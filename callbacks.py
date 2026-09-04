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

    def __init__(self, video_freq=500, video_max_steps=20000, video_fps=8,
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
        self.episode_timeouts = []
        self.episode_offroute = []
        self.frontier_cells = None
        self._last_logged_epoch = -1
        self.door_x = []          # max_x of NON-restart (from-door) episodes
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
            # 0xFF is the game-over sentinel, not 255 lives
            self.episode_lives.append(0 if info['life'] == 255 else info['life'])

        if 'start_stage' in info:
            # A clear is an ON-ROUTE level advance (stages_cleared) or an
            # outright victory. progress_gain also counts OFF-ROUTE exits
            # (1-2 -> 1-3), which made the curriculum starve exactly the
            # level whose wrong exit the policy was taking.
            self.stage_records.setdefault(info['start_stage'], []).append(
                (float(info.get('progress_gain', 0)),
                 float(info.get('warped', False)),
                 float(info.get('victory', False)),
                 float(info.get('stages_cleared', 0)),
                 float(not info.get('self_restart', False))))
        if 'victory' in info:
            self.episode_victories.append(float(info['victory']))
        if 'looped' in info:
            self.episode_loops.append(float(info['looped']))
        if 'self_restart' in info and not info['self_restart']:
            self.door_x.append(info.get('max_x_pos', 0))
        if 'idle_timeout' in info:
            self.episode_timeouts.append(float(info['idle_timeout']))
        if 'offroute' in info:
            self.episode_offroute.append(float(info['offroute']))
        if 'frontier_cells' in info:
            self.frontier_cells = info['frontier_cells']

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
        # rl_games calls this twice per epoch (write_stats + train loop);
        # the second call would republish the non-buffered scalars
        if epoch_num == self._last_logged_epoch:
            return
        self._last_logged_epoch = epoch_num

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
            # columns: gain, warped, victory, stages_cleared, is_door
            arr = np.array(recs)
            cleared = (arr[:, 3] > 0) | (arr[:, 2] > 0)
            self.writer.add_scalar(f'mario/gain/{stage}',
                                   float(arr[:, 0].mean()), epoch_num)
            self.writer.add_scalar(f'mario/clear/{stage}',
                                   float(cleared.mean()), epoch_num)
            self.writer.add_scalar(f'mario/warp/{stage}',
                                   float(arr[:, 1].mean()), epoch_num)
            # the curriculum re-weights DOOR resets, so its signal must come
            # from door episodes (archive restarts start mid-level and would
            # make a level look mastered)
            door = arr[:, 4] > 0
            if door.any():
                dclear = float(cleared[door].mean())
                self.writer.add_scalar(f'mario/clear_door/{stage}', dclear,
                                       epoch_num)
                prev = self._clear_ema.get(stage, 0.0)
                self._clear_ema[stage] = prev + 0.1 * (dclear - prev)

        if len(self.episode_victories) > 0:
            self.writer.add_scalar('mario/victory_rate',
                                   float(np.mean(self.episode_victories)),
                                   epoch_num)
        if len(self.episode_loops) > 0:
            self.writer.add_scalar('mario/loop_rate',
                                   float(np.mean(self.episode_loops)),
                                   epoch_num)
        if self.frontier_cells is not None:
            self.writer.add_scalar('mario/frontier_cells',
                                   self.frontier_cells, epoch_num)
        if len(self.episode_offroute) > 0:
            self.writer.add_scalar('mario/offroute_rate',
                                   float(np.mean(self.episode_offroute)),
                                   epoch_num)
        if len(self.episode_timeouts) > 0:
            self.writer.add_scalar('mario/idle_timeout_rate',
                                   float(np.mean(self.episode_timeouts)),
                                   epoch_num)
        if len(self.door_x) > 0:
            self.writer.add_scalar('mario/door_max_x',
                                   float(np.max(self.door_x)), epoch_num)
            self.writer.add_scalar('mario/door_mean_x',
                                   float(np.mean(self.door_x)), epoch_num)

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
        self.episode_timeouts.clear()
        self.episode_offroute.clear()
        self.frontier_cells = None
        self.door_x.clear()

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

    def _make_eval_env(self, **overrides):
        """Build the env used for video recording (sequential full game from
        1-1 by default; eval_env_kwargs overrides, e.g. start stage;
        `overrides` win over both, e.g. random_stages=[level])."""
        from mario_env import create_mario_env
        kwargs = dict(
            name='SuperMarioBros-v0',
            action_type='complex',
            episode_life=False,
            skip=4,
            sticky_actions=0.0,
        )
        # reward semantics must match training so the video's R readout is
        # the trained signal (idle default would silently be 150; loop and
        # fail penalties default to 0/15 and made loops look free)
        env_cfg = self.algo.env_config or {}
        for k in ('idle_timeout', 'idle_penalty', 'idle_threshold',
                  'loop_penalty', 'fail_penalty', 'backtrack_penalty',
                  'progress_reward', 'score_reward', 'x_reward', 'obs_mode',
                  'stage_bonus', 'offroute_penalty', 'novelty_bonus',
                  'novelty_global', 'novelty_y_band', 'reward'):
            if k in env_cfg:
                kwargs[k] = env_cfg[k]
        # the on-route set follows the CONFIG, never the clip's start level:
        # a per-level clip passes random_stages=[lvl], which would otherwise
        # make every legal advance (including the 1-2 warp) off-route
        route = env_cfg.get('route_levels') or env_cfg.get('random_stages')
        if route:
            kwargs['route_levels'] = list(route)
        kwargs.update(self.eval_env_kwargs or {})
        kwargs.pop('video_levels', None)        # recorder options, not env
        kwargs.pop('video_level_steps', None)
        kwargs.update(overrides)
        backend = kwargs.pop('backend', 'retro')
        if backend in ('native', 'lockstep'):
            from mario_native_vecenv import NativeEvalEnv, LockstepVideoEnv
            kwargs.pop('name', None)
            kwargs.pop('action_type', None)
            kwargs.pop('record_frames', None)
            cls = LockstepVideoEnv if backend == 'lockstep' else NativeEvalEnv
            return cls(**kwargs)
        return create_mario_env(**kwargs)

    @staticmethod
    def draw_strip(frame, epoch_num, stat, font):
        """Frame + 2-line stats strip below it (gameplay pixels untouched):
        line 1 = epoch, level, x, cumulative R, this step's reward;
        line 2 = reward event flash (LOOP/DEATH/OFF-ROUTE/IDLE/CLEAR)."""
        from PIL import Image, ImageDraw
        world, stage, x_pos, rew, r_step, event, lives = stat
        bar_h = 32                      # 224 + 32 = 256: codec-friendly height
        img = Image.fromarray(frame)
        canvas = Image.new('RGB', (img.width, img.height + bar_h), (0, 0, 0))
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, img.height + 2),
                  f'ep {epoch_num}  {world}-{stage}  x={x_pos}  L={lives}  '
                  f'R={rew:.0f}  r={r_step:+.1f}',
                  fill=(255, 255, 255), font=font)
        if event:
            color = (80, 255, 80) if event.startswith('CLEAR') \
                else (255, 80, 80)
            draw.text((4, img.height + 17), event, fill=color, font=font)
        return canvas

    def _play_clip(self, model, env, max_steps, epoch_num,
                   stop_on_level_change=False):
        try:
            return self._play_clip_inner(model, env, max_steps, epoch_num,
                                         stop_on_level_change)
        finally:
            # a leaked stable-retro emulator makes every later video fail
            env.close()

    def _play_clip_inner(self, model, env, max_steps, epoch_num,
                         stop_on_level_change=False):
        """Play one clip with the sampled policy until the episode is over
        (max_steps is only a safety cap). With stop_on_level_change the clip
        also ends once the level is cleared (per-level clips).

        Returns (raw_frames, pil_frames_with_strip, step_stats, last_info,
        total_reward, frames_per_step). Closes the env."""
        from PIL import ImageFont
        frames, step_stats = [], []
        obs = env.reset()
        total_reward, info = 0, {}
        gp0 = None
        prev_life, event, event_ttl = None, '', 0
        per_step_frames = getattr(env.unwrapped, 'frames_per_step',
                                  4 if hasattr(env.unwrapped, 'frames4')
                                  else 1)
        is_rnn = self.algo.is_rnn
        if is_rnn:
            # Default state is (num_layers, num_actors, hidden)
            # For single-env eval we need (num_layers, 1, hidden)
            rnn_states = model.get_default_rnn_state()
            rnn_states = [s[:, :1, :].contiguous() for s in rnn_states]
        for step in range(max_steps):
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                input_dict = {'obs': obs_tensor, 'is_train': False}
                if is_rnn:
                    input_dict['rnn_states'] = rnn_states
                    input_dict['seq_length'] = 1
                res = model(input_dict)
            if is_rnn:
                rnn_states = res.get('rnn_states', rnn_states)
            # sample, don't argmax: the stochastic policy is what PPO
            # optimizes; argmax has fixed points (same frame -> same
            # action, e.g. running into a stair block forever) that the
            # trained policy never exhibits
            action = torch.distributions.Categorical(
                logits=res['logits']).sample().item()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            # reward events, flashed on the strip for ~1s so penalties
            # are auditable from the video (R alone hides a -100 that
            # lands on the same step as a +8)
            life = info.get('life', prev_life)
            if info.get('looped'):
                event, event_ttl = 'LOOP %+.0f' % reward, 60
            elif info.get('offroute'):
                event, event_ttl = 'OFF-ROUTE %+.0f' % reward, 60
            elif info.get('idle_timeout'):
                event, event_ttl = 'IDLE TIMEOUT %+.0f' % reward, 60
            elif life == 255 and prev_life is not None and prev_life != 255:
                event, event_ttl = 'GAME OVER %+.0f' % reward, 60
            elif prev_life is not None and life < prev_life:
                event, event_ttl = 'DEATH %+.0f' % reward, 60
            elif info.get('flag_get') or info.get('victory'):
                event, event_ttl = 'CLEAR %+.0f' % reward, 60
            prev_life = life
            stat = (info.get('world', 1), info.get('stage', 1),
                    info.get('x_pos', 0), total_reward, reward,
                    event if event_ttl > 0 else '',
                    (life if life != 255 else 0))
            event_ttl -= per_step_frames
            if hasattr(env.unwrapped, 'frames4'):
                # native eval: all 4 emulated frames -> no aliasing
                for f in env.unwrapped.frames4:
                    frames.append(f)
                    step_stats.append(stat)
            else:
                frames.append(env.unwrapped.screen.copy())
                step_stats.append(stat)
            gp = info.get('game_progress')
            if gp0 is None:
                gp0 = gp
            if done or (stop_on_level_change and gp is not None
                        and gp != gp0 and step > 8):  # noqa: E501
                # hold the terminal frame ~1s so the event flash (LOOP /
                # DEATH / GAME OVER ...) is actually visible, not 4 frames
                for _ in range(60):
                    frames.append(frames[-1])
                    step_stats.append(stat)
                break
        cause = ('victory' if info.get('victory') else 'loop' if info.get('looped')
                 else 'idle timeout' if info.get('idle_timeout') else 'off-route'
                 if info.get('offroute') else 'game over' if info.get('life') == 255
                 else 'level cleared' if (gp0 is not None and info.get('game_progress') != gp0)
                 else 'step cap')
        print(f'  [Video] clip {info.get("world", "?")}-{info.get("stage", "?")}: '
              f'{len(frames)} frames, ended by {cause} at x={info.get("x_pos", 0)}')
        font = ImageFont.load_default()
        pil_frames = [self.draw_strip(f, epoch_num, s, font)
                      for s, f in zip(step_stats, frames)]
        return frames, pil_frames, step_stats, info, total_reward, per_step_frames

    @staticmethod
    def _gif_bytes(pil_frames, per_step, every=None):
        """Encode frames as an animated GIF. Delays are centisecond-
        quantized: at 60fps material use every 2nd frame with a 40/30/30ms
        cycle (33.3ms avg), every 4th with 70/70/60 (66.7ms); at 15fps
        material use all frames with the 70/70/60 cycle."""
        import tempfile
        if every is None:
            every = 2 if per_step >= 2 else 1
            # long clips: bound the GIF (TB) at ~3000 frames by halving the
            # rate again; the mp4 on disk keeps every frame
            if len(pil_frames) // every > 3000:
                every *= 2
        gif_frames = pil_frames[::every]
        fast = per_step >= 2 and every <= 2
        durations = [(40 if i % 3 == 2 else 30) if fast else
                     (60 if i % 3 == 2 else 70) for i in range(len(gif_frames))]
        path = tempfile.NamedTemporaryFile(suffix='.gif', delete=False).name
        gif_frames[0].save(path, save_all=True, append_images=gif_frames[1:],
                           duration=durations, loop=0, optimize=True)
        with open(path, 'rb') as f:
            data = f.read()
        os.remove(path)
        return data

    def _clean_door_eval(self, model, epoch_num, n=32, max_steps=600):
        """Door episodes WITHOUT exploration noise (no sticky/random actions,
        no restarts): what the policy can do vs what the noise does to it.
        Training door metrics run under noise (8-4: 42% of door episodes
        died in the opening lava with 5% random + 10% sticky actions while
        the clean policy entered pipe 1 61/64 times)."""
        from mario_native_vecenv import MarioNativeVecEnv
        ec = dict(self.algo.env_config or {})
        for k in ('explore_eps', 'sticky_actions', 'self_restart_prob',
                  'explore_episode_prob', 'novelty_bonus'):
            ec[k] = 0
        for k in ('name', 'action_type', 'archive_path'):
            ec.pop(k, None)
        ec['n_threads'] = 4
        env = MarioNativeVecEnv('clean', n, dense_infos=False, **ec)
        try:
            obs = env.reset()
            max_x = np.zeros(n); fin = {}
            for step in range(max_steps):
                with torch.no_grad():
                    res = model({'obs': torch.from_numpy(obs).float(),
                                 'is_train': False})
                a = torch.distributions.Categorical(
                    logits=res['logits']).sample().numpy()
                obs, r, d, infos = env.step(a)
                for i in range(n):
                    if i in fin:
                        continue
                    max_x[i] = max(max_x[i], env.max_x[i])
                    if d[i]:
                        inf = infos[i] if isinstance(infos, list) else {}
                        fin[i] = ('victory' if inf.get('victory') else
                                  'loop' if inf.get('looped') else
                                  'idle' if inf.get('idle_timeout') else
                                  'offroute' if inf.get('offroute') else
                                  'death')
                if len(fin) == n:
                    break
            for i in range(n):
                fin.setdefault(i, 'running')
            counts = {k: sum(1 for v in fin.values() if v == k) / n
                      for k in ('death', 'loop', 'victory', 'idle', 'offroute',
                                'running')}
            self.writer.add_scalar('eval/door_max_x_mean', float(max_x.mean()),
                                   epoch_num)
            self.writer.add_scalar('eval/door_max_x_max', float(max_x.max()),
                                   epoch_num)
            for k, v in counts.items():
                self.writer.add_scalar(f'eval/door_{k}_rate', v, epoch_num)
            print(f'  [Eval] clean door x{n}: max_x mean {max_x.mean():.0f} '
                  f'max {max_x.max():.0f} | ' + ' '.join(
                      f'{k} {v:.2f}' for k, v in counts.items()))
        finally:
            env.close()

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
                self._clean_door_eval(model, epoch_num)
            except Exception as e:
                print(f'  [Eval] clean door eval failed: {e}')
            try:
                from tensorboardX.proto.summary_pb2 import Summary
            except ImportError:
                from tensorboard.compat.proto.summary_pb2 import Summary

            # main clip: sequential game from the eval start (1-1 by default)
            (frames, pil_frames, step_stats, info, total_reward,
             per_step) = self._play_clip(model, self._make_eval_env(),
                                         self.video_max_steps, epoch_num)

            if len(frames) > 4:
                run_dir = os.path.dirname(os.path.dirname(
                    self.writer.file_writer.event_writer._ev_writer._file_name))
                video_dir = os.path.join(run_dir, 'videos')
                os.makedirs(video_dir, exist_ok=True)
                mp4_path = os.path.join(video_dir, f'epoch_{epoch_num}.mp4')
                # mp4 real-time rate: skip frames per step, 60Hz game
                imageio.mimsave(mp4_path, [np.asarray(c) for c in pil_frames],
                                fps=60 if per_step >= 2 else 15)

                # GIF for the TB Images tab
                gif_bytes = self._gif_bytes(pil_frames, per_step)

                w, h = pil_frames[0].size
                summary = Summary(value=[Summary.Value(
                    tag='gameplay/agent',
                    image=Summary.Image(
                        height=h, width=w, colorspace=3,
                        encoded_image_string=gif_bytes),
                )])
                self.writer.file_writer.add_summary(summary, epoch_num)
                self.writer.flush()

                # per-level clips (multi-level training): one clip from each
                # level's door -> mp4 each + one synchronized mosaic GIF, so
                # the policy's play on every level can be compared at once
                ek = self.eval_env_kwargs or {}
                levels = ek.get('video_levels')
                if levels is None:
                    rs = (self.algo.env_config or {}).get('random_stages')
                    levels = list(rs) if rs and len(rs) > 1 else []
                if levels:
                    # play until the level is cleared or the lives are gone
                    # (video_level_steps is only an optional safety cap)
                    lvl_steps = int(ek.get('video_level_steps')
                                    or self.video_max_steps)
                    sizes = []
                    for lvl in levels:
                        env_l = self._make_eval_env(random_stages=[lvl],
                                                    full_game=True)
                        fr, pf, st, inf_l, rew_l, ps = self._play_clip(
                            model, env_l, lvl_steps, epoch_num,
                            stop_on_level_change=True)
                        if len(fr) < 4:
                            continue
                        imageio.mimsave(
                            os.path.join(video_dir,
                                         f'epoch_{epoch_num}_{lvl}.mp4'),
                            [np.asarray(c) for c in pf],
                            fps=60 if ps >= 2 else 15)
                        gb = self._gif_bytes(pf, ps)
                        gw, gh = pf[0].size
                        self.writer.file_writer.add_summary(Summary(value=[
                            Summary.Value(tag=f'gameplay/level_{lvl}',
                                          image=Summary.Image(
                                              height=gh, width=gw,
                                              colorspace=3,
                                              encoded_image_string=gb))]),
                            epoch_num)
                        self.writer.add_scalar(f'eval/level_max_x/{lvl}',
                                               max(s[2] for s in st),
                                               epoch_num)
                        sizes.append(len(gb) // 1024)
                    self.writer.flush()
                    print(f'  [Video] Epoch {epoch_num}: {len(sizes)} level '
                          f'clips, GIF KB {sizes}')

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
