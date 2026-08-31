"""Custom AlgoObserver for Mario training with TensorBoard video logging."""

import copy
import os
import threading

import numpy as np
import torch
from rl_games.common.algo_observer import AlgoObserver
from rl_games.algos_torch import torch_ext


class MarioObserver(AlgoObserver):
    """Logs Mario-specific metrics and records gameplay videos to TensorBoard.

    Tracked metrics:
        - game_progress: furthest stage reached (0-31 across worlds 1-8)
        - max_x_pos: furthest x position reached in an episode
        - flag_get: whether Mario completed a stage
        - lives: lives remaining at episode end

    Video recording:
        - Records agent gameplay every `video_freq` epochs
        - Videos show the full game (no episode_life) with deterministic policy
    """

    def __init__(self, video_freq=500, video_max_steps=4000, video_fps=8,
                 curriculum_freq=0, eval_env_kwargs=None):
        super().__init__()
        self.video_freq = video_freq
        self.video_max_steps = video_max_steps
        self.video_fps = video_fps
        # Overrides for the video/eval env (e.g. start at the trained stage
        # instead of the sequential 1-1 game)
        self.eval_env_kwargs = eval_env_kwargs
        # Every N epochs, re-weight random-stage sampling toward stages with
        # low clear rate (0 disables).
        self.curriculum_freq = curriculum_freq

        # Metrics buffers (collected across episodes within an epoch)
        self.episode_x_pos = []
        self.episode_progress = []
        self.episode_flags = []
        self.episode_lives = []
        # start_stage -> (progress_gain, warped, victory) per finished episode
        self.stage_records = {}
        self.episode_victories = []
        self.episode_loops = []
        self._clear_ema = {}  # start_stage -> EMA of clear rate

        self.best_progress = 0
        self.best_x_pos = 0

        self._video_thread = None

    def after_init(self, algo):
        self.algo = algo
        self.writer = algo.writer
        self.stage_list = (algo.env_config or {}).get('random_stages')
        self.game_scores = torch_ext.AverageMeter(
            1, self.algo.games_to_track).to(self.algo.ppo_device)

    def process_infos(self, infos, done_indices):
        """Collect Mario-specific info from completed episodes."""
        if not infos:
            return

        done_indices = done_indices.cpu().numpy()

        if not isinstance(infos, dict) and len(infos) > 0 and isinstance(infos[0], dict):
            for ind in done_indices:
                ind = ind.item()
                if len(infos) <= ind // self.algo.num_agents:
                    continue
                info = infos[ind // self.algo.num_agents]
                self._process_single_info(info)
        elif isinstance(infos, dict):
            for ind in done_indices:
                self._process_single_info(infos)

    def _process_single_info(self, info):
        if 'max_x_pos' in info:
            self.episode_x_pos.append(info['max_x_pos'])
        elif 'x_pos' in info:
            self.episode_x_pos.append(info['x_pos'])

        if 'game_progress' in info:
            self.episode_progress.append(info['game_progress'])

        if 'flag_get' in info:
            self.episode_flags.append(float(info['flag_get']))

        if 'life' in info:
            self.episode_lives.append(info['life'])

        if 'start_stage' in info:
            # A clear is progress gained OR outright victory (beating 8-4
            # cannot increase game_progress past 31)
            self.stage_records.setdefault(info['start_stage'], []).append(
                (float(info.get('progress_gain', 0)),
                 float(info.get('warped', False)),
                 float(info.get('victory', False))))
        if 'victory' in info:
            self.episode_victories.append(float(info['victory']))
        if 'looped' in info:
            self.episode_loops.append(float(info['looped']))

        # Also track game scores for the default scorer
        game_res = info.get('scores', None)
        if game_res is not None:
            self.game_scores.update(
                torch.from_numpy(np.asarray([game_res])).to(self.algo.ppo_device))

    def after_clear_stats(self):
        self.game_scores.clear()

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.writer is None:
            return

        # Default scores
        if self.game_scores.current_size > 0:
            mean_scores = self.game_scores.get_mean()
            self.writer.add_scalar('scores/mean', mean_scores, frame)
            self.writer.add_scalar('scores/iter', mean_scores, epoch_num)

        # Mario-specific metrics
        if len(self.episode_x_pos) > 0:
            mean_x = np.mean(self.episode_x_pos)
            max_x = np.max(self.episode_x_pos)
            self.writer.add_scalar('mario/mean_x_pos', mean_x, epoch_num)
            self.writer.add_scalar('mario/max_x_pos', max_x, epoch_num)
            if max_x > self.best_x_pos:
                self.best_x_pos = max_x
            self.writer.add_scalar('mario/best_x_pos', self.best_x_pos, epoch_num)

        if len(self.episode_progress) > 0:
            mean_prog = np.mean(self.episode_progress)
            max_prog = np.max(self.episode_progress)
            self.writer.add_scalar('mario/mean_stage_progress', mean_prog, epoch_num)
            self.writer.add_scalar('mario/max_stage_progress', max_prog, epoch_num)
            if max_prog > self.best_progress:
                self.best_progress = max_prog
                world = int(self.best_progress // 4) + 1
                stage = int(self.best_progress % 4) + 1
                print(f'  [Mario] New best progress: World {world}-{stage}')
            self.writer.add_scalar('mario/best_stage_progress', self.best_progress, epoch_num)

        if len(self.episode_flags) > 0:
            flag_rate = np.mean(self.episode_flags)
            self.writer.add_scalar('mario/flag_get_rate', flag_rate, epoch_num)

        if len(self.episode_lives) > 0:
            mean_lives = np.mean(self.episode_lives)
            self.writer.add_scalar('mario/mean_lives_remaining', mean_lives, epoch_num)

        # Per-start-stage metrics + clear-rate EMA (drives the curriculum)
        for stage, recs in self.stage_records.items():
            arr = np.array(recs)  # columns: gain, warped, victory
            clear = float(np.mean((arr[:, 0] >= 1) | (arr[:, 2] > 0)))
            self.writer.add_scalar(f'mario/gain/{stage}',
                                   float(arr[:, 0].mean()), epoch_num)
            self.writer.add_scalar(f'mario/clear/{stage}', clear, epoch_num)
            self.writer.add_scalar(f'mario/warp/{stage}',
                                   float(arr[:, 1].mean()), epoch_num)
            prev = self._clear_ema.get(stage, 0.0)
            self._clear_ema[stage] = prev + 0.1 * (clear - prev)

        if len(self.episode_victories) > 0:
            self.writer.add_scalar('mario/victory_rate',
                                   float(np.mean(self.episode_victories)),
                                   epoch_num)
        if len(self.episode_loops) > 0:
            self.writer.add_scalar('mario/loop_rate',
                                   float(np.mean(self.episode_loops)),
                                   epoch_num)

        # Curriculum: sample unmastered stages more often
        if (self.curriculum_freq > 0 and epoch_num % self.curriculum_freq == 0
                and self.stage_list
                and hasattr(getattr(self.algo, 'vec_env', None),
                            'set_stage_weights')):
            weights = {s: 0.15 + (1.0 - self._clear_ema.get(s, 0.0))
                       for s in self.stage_list}
            self.algo.vec_env.set_stage_weights(weights)
            for s, w in weights.items():
                self.writer.add_scalar(f'mario/weight/{s}', w, epoch_num)

        # Clear buffers
        self.episode_x_pos.clear()
        self.episode_progress.clear()
        self.episode_flags.clear()
        self.episode_lives.clear()
        self.stage_records.clear()
        self.episode_victories.clear()
        self.episode_loops.clear()

        # Record video periodically, on a background thread so training never
        # blocks. The thread gets a CPU copy of the model: no GPU access, and
        # the live model keeps training undisturbed.
        if self.video_freq > 0 and epoch_num % self.video_freq == 0 and epoch_num > 0:
            if self._video_thread is not None and self._video_thread.is_alive():
                print(f'  [Video] Epoch {epoch_num}: skipped, previous recording '
                      f'still in progress')
            else:
                # Copy the eager module, never the torch.compile wrapper:
                # calling a compiled copy triggers dynamo tracing whose FX
                # patching is process-global and crashes the training thread.
                eager_model = getattr(self.algo.model, '_orig_mod', self.algo.model)
                model_copy = copy.deepcopy(eager_model).to('cpu')
                model_copy.eval()
                self._video_thread = threading.Thread(
                    target=self._record_video, args=(epoch_num, model_copy),
                    daemon=True)
                self._video_thread.start()

    def _make_eval_env(self):
        """Build the env used for video recording (sequential full game from
        1-1 by default; eval_env_kwargs overrides, e.g. start stage)."""
        from mario_env import create_mario_env
        kwargs = dict(
            name='SuperMarioBros-v0',
            action_type='complex',
            episode_life=False,
            stage_bonus=0,
            skip=4,
            sticky_actions=0.0,
        )
        kwargs.update(self.eval_env_kwargs or {})
        if kwargs.pop('backend', 'retro') == 'native':
            from mario_native_vecenv import NativeEvalEnv
            kwargs.pop('name', None)
            kwargs.pop('action_type', None)
            return NativeEvalEnv(**kwargs)
        return create_mario_env(**kwargs)

    def _record_video(self, epoch_num, model):
        """Record gameplay: PIL GIF to TensorBoard + MP4 to disk.

        Runs on a background thread with a CPU-only copy of the model, so
        training continues undisturbed and no GPU is touched from the thread.
        """
        try:
            import imageio
            import tempfile
            from PIL import Image, ImageDraw, ImageFont
            try:
                from tensorboardX.proto.summary_pb2 import Summary
            except ImportError:
                from tensorboard.compat.proto.summary_pb2 import Summary

            env = self._make_eval_env()

            frames = []
            step_stats = []
            obs = env.reset()
            done = False
            total_reward = 0
            info = {}

            is_rnn = self.algo.is_rnn
            if is_rnn:
                # Default state is (num_layers, num_actors, hidden)
                # For single-env eval we need (num_layers, 1, hidden)
                rnn_states = model.get_default_rnn_state()
                rnn_states = [s[:, :1, :].contiguous() for s in rnn_states]

            for step in range(self.video_max_steps):
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)

                with torch.no_grad():
                    input_dict = {
                        'obs': obs_tensor,
                        'is_train': False,
                    }
                    if is_rnn:
                        input_dict['rnn_states'] = rnn_states
                        input_dict['seq_length'] = 1
                    res = model(input_dict)

                if is_rnn:
                    rnn_states = res.get('rnn_states', rnn_states)

                action = torch.argmax(res['logits'], dim=-1).item()
                obs, reward, done, info = env.step(action)
                frames.append(env.unwrapped.screen.copy())
                total_reward += reward
                step_stats.append((info.get('world', 1), info.get('stage', 1),
                                   info.get('x_pos', 0), total_reward))
                if done:
                    break

            env.close()

            if len(frames) > 4:
                # Stats strip below the frame: gameplay pixels stay untouched.
                # 240 + 16 = 256 high, which video encoders also prefer.
                font = ImageFont.load_default()
                bar_h = 16
                pil_frames = []
                for (world, stage, x_pos, rew), f in zip(step_stats, frames):
                    img = Image.fromarray(f)
                    canvas = Image.new('RGB', (img.width, img.height + bar_h),
                                       (0, 0, 0))
                    canvas.paste(img, (0, 0))
                    draw = ImageDraw.Draw(canvas)
                    draw.text((4, img.height + 2),
                              f'ep {epoch_num}  {world}-{stage}  '
                              f'x={x_pos}  R={rew:.0f}',
                              fill=(255, 255, 255), font=font)
                    pil_frames.append(canvas)

                # Save MP4 to disk (15 fps = exactly the 66.67ms real frame
                # time: 4 NES frames at 60fps)
                run_dir = os.path.dirname(os.path.dirname(
                    self.writer.file_writer.event_writer._ev_writer._file_name))
                video_dir = os.path.join(run_dir, 'videos')
                os.makedirs(video_dir, exist_ok=True)
                mp4_path = os.path.join(video_dir, f'epoch_{epoch_num}.mp4')
                imageio.mimsave(mp4_path, [np.asarray(c) for c in pil_frames],
                                fps=15)

                # Write PIL animated GIF to TensorBoard Images tab. GIF delays
                # are centisecond-quantized, so a constant 67ms is impossible;
                # a 70/70/60 cycle averages exactly 66.67ms (real time).
                durations = [60 if i % 3 == 2 else 70
                             for i in range(len(pil_frames))]
                gif_path = tempfile.NamedTemporaryFile(suffix='.gif', delete=False).name
                pil_frames[0].save(gif_path, save_all=True,
                                   append_images=pil_frames[1:],
                                   duration=durations,
                                   loop=0, optimize=True)
                with open(gif_path, 'rb') as f:
                    gif_bytes = f.read()
                os.remove(gif_path)

                w, h = pil_frames[0].size
                summary = Summary(value=[Summary.Value(
                    tag='gameplay/agent',
                    image=Summary.Image(
                        height=h, width=w, colorspace=3,
                        encoded_image_string=gif_bytes),
                )])
                self.writer.file_writer.add_summary(summary, epoch_num)
                self.writer.flush()

                x_pos = info.get('x_pos', 0)
                world = info.get('world', 1)
                stage = info.get('stage', 1)
                # Eval frontier scalars: deterministic sequential run from 1-1
                self.writer.add_scalar('eval/game_progress',
                                       info.get('game_progress', 0), epoch_num)
                self.writer.add_scalar(
                    'eval/max_x', max(s[2] for s in step_stats), epoch_num)
                print(f'  [Video] Epoch {epoch_num}: reward={total_reward:.0f}, '
                      f'world={world}-{stage}, x_pos={x_pos}, '
                      f'gif={len(gif_bytes)/1024:.0f}KB, mp4={mp4_path}')

        except Exception as e:
            import traceback
            print(f'  [Video] Recording failed: {e}')
            traceback.print_exc()
