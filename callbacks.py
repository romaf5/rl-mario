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
                 curriculum_freq=0, eval_env_kwargs=None, eval_episodes=32,
                 eval_level_steps=1500, eval_seq_episodes=8, eval_seq_steps=6000):
        super().__init__()
        self.video_freq = video_freq
        self.video_max_steps = video_max_steps
        self.video_fps = video_fps
        # sampled-policy evaluation sizes (per-level door episodes and
        # sequential full-game episodes) -- see _level_eval / _sequential_eval
        self.eval_episodes = eval_episodes
        self.eval_level_steps = eval_level_steps
        self.eval_seq_episodes = eval_seq_episodes
        self.eval_seq_steps = eval_seq_steps
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
        self.episode_loop_timeouts = []
        self.episode_gaps = []
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
            # the flag byte is already clear on the step the episode ends
            # (level change), so the rate read 0.000 forever; count the
            # level advance instead
            self.episode_flags.append(float(bool(info['flag_get'])
                                            or info.get('stages_cleared', 0) > 0))

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
        if 'page_resets' in info:
            self.episode_loops.append(float(info['page_resets'] > 0))
        if 'loop_timeout' in info:
            self.episode_loop_timeouts.append(float(info['loop_timeout']))
        if 'max_unpaid_gap' in info:
            self.episode_gaps.append(float(info['max_unpaid_gap']))
        if 'self_restart' in info and not info['self_restart']:
            self.door_x.append(info.get('max_x_pos', 0))
        if 'timeout' in info:
            self.episode_timeouts.append(float(info['timeout']))
        if 'wrong_exit' in info:
            self.episode_offroute.append(float(info['wrong_exit']))
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
            # NB in full-game mode the flag is not a terminal, so this only
            # catches episodes that happened to end on a flag frame; the
            # meaningful signal is mario/clear/<level> and the rate below
            self.writer.add_scalar('mario/flag_get_rate',
                                   float(np.mean(self.episode_flags)),
                                   epoch_num)
        if self.stage_records:
            tot = [r for recs in self.stage_records.values() for r in recs]
            self.writer.add_scalar(
                'mario/level_clear_rate',
                float(np.mean([(r[3] > 0) or (r[2] > 0) for r in tot])),
                epoch_num)

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
            self.writer.add_scalar('mario/page_reset_rate',
                                   float(np.mean(self.episode_loops)),
                                   epoch_num)
        if self.frontier_cells is not None:
            self.writer.add_scalar('mario/frontier_cells',
                                   self.frontier_cells, epoch_num)
        if len(self.episode_offroute) > 0:
            self.writer.add_scalar('mario/wrong_exit_rate',
                                   float(np.mean(self.episode_offroute)),
                                   epoch_num)
        if len(self.episode_timeouts) > 0:
            self.writer.add_scalar('mario/timeout_rate',
                                   float(np.mean(self.episode_timeouts)),
                                   epoch_num)
        if len(self.episode_loop_timeouts) > 0:
            self.writer.add_scalar('mario/loop_timeout_rate',
                                   float(np.mean(self.episode_loop_timeouts)),
                                   epoch_num)
        if len(self.episode_gaps) > 0:
            # longest unpaid stretch per episode: tells whether the cutoff
            # (unpaid_timeout) is cutting real play or genuinely stuck runs
            self.writer.add_scalar('mario/max_unpaid_gap_mean',
                                   float(np.mean(self.episode_gaps)), epoch_num)
            self.writer.add_scalar('mario/max_unpaid_gap_p95',
                                   float(np.percentile(self.episode_gaps, 95)),
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
        self.episode_loop_timeouts.clear()
        self.episode_gaps.clear()
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
        for k in ('unpaid_timeout', 'page_reset_grace', 'page_reset_px',
                  'stage_bonus', 'reward'):
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
        # default to the training backend: silently falling back to retro
        # would record a different observation distribution
        native_cfg = any(k in env_cfg for k in ('archive_path', 'unpaid_timeout',
                                                 'self_restart_prob', 'cell_x_bin'))
        backend = kwargs.pop('backend', 'native' if native_cfg else 'retro')
        raw_steps = kwargs.pop('raw_steps', True)
        if backend == 'lockstep':
            # retired: the policy played on the native core while a
            # stable-retro emulator rendered the frames, and the eval's timer
            # hack (forced time-up on a stuck life) only reached the native
            # core -- every clip diverged from the game after the first
            # stuck event (2026-09-11 review)
            print('  [Video] backend "lockstep" is retired; recording on the '
                  'native core (hack-free)')
            backend = 'native'
        if backend == 'native':
            from mario_native_vecenv import NativeEvalEnv
            kwargs.pop('name', None)
            kwargs.pop('action_type', None)
            kwargs.pop('record_frames', None)
            return NativeEvalEnv(raw_steps=raw_steps, **kwargs)
        return create_mario_env(**kwargs)

    @staticmethod
    def draw_strip(frame, epoch_num, stat, font):
        """Frame + 2-line stats strip below it (gameplay pixels untouched):
        line 1 = epoch, level, x, this LIFE's cumulative R, this step's reward;
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
                   stop_on_level_change=False, seed=0):
        try:
            return self._play_clip_inner(model, env, max_steps, epoch_num,
                                         stop_on_level_change, seed)
        finally:
            # a leaked stable-retro emulator makes every later video fail
            env.close()

    def _play_clip_inner(self, model, env, max_steps, epoch_num,
                         stop_on_level_change=False, seed=0):
        """Play one clip with the SAMPLED policy (local generator seeded
        with `seed`: reproducible per epoch, and it is the policy PPO
        trains -- argmax froze on fixed points such as holding right+A into
        a staircase step for 300 steps) until the episode is over (max_steps
        is only a safety cap). With stop_on_level_change the clip also ends
        once the level is cleared (per-level clips).

        Returns (raw_frames, pil_frames_with_strip, step_stats, last_info,
        total_reward, frames_per_step). Closes the env."""
        from PIL import ImageFont
        frames, step_stats = [], []
        obs = env.reset()
        total_reward, info = 0, {}
        life_reward = 0.0       # the strip shows THIS life's reward: a sum
        gp0 = None              # across lives grew with every death
        # sidecar trace of the clip: start state + actions + per-term rewards
        # + flags, replayable with tools/play.py --replay <file>.npz
        tr_env = getattr(env.unwrapped, 'v', None)
        tr_state, tr_rows, tr_acts = None, [], []
        tr_start_lvl = None
        if tr_env is not None:
            tr_env.lib.benv_save(tr_env.env, 0, tr_env._sbuf)
            tr_state = bytes(tr_env._sbuf.raw)
            tr_env._fetch_obs(0); r0 = tr_env.ram[0]
            tr_start_lvl = '%d-%d' % (int(r0[0x75F]) + 1, int(r0[0x75C]) + 1)   # trace files are named by the START level
        prev_life, event, event_ttl = None, '', 0
        gen = torch.Generator().manual_seed(int(seed))
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
            action = int(self._sample_actions(res['logits'], gen)[0])
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if prev_life is not None and info.get('life', prev_life) != prev_life:
                life_reward = 0.0
            life_reward += reward
            if tr_env is not None:
                sg = tr_env.last_signals
                tr_acts.append(int(action))
                tr_rows.append([step, int(action), info.get('x_pos', 0),
                                int(sg.ypix[0]), info.get('life', -1),
                                round(float(reward), 3)]
                               + [round(float(v[0]), 3) for v in tr_env.last_terms.values()]
                               + [int(sg.page_reset[0]), int(sg.timeout[0]),
                                  int(sg.died[0]), int(sg.frame_change[0]),
                                  int(sg.level_delta[0] > 0), int(done)])
            # reward events, flashed on the strip for ~1s so penalties
            # are auditable from the video (R alone hides a -100 that
            # lands on the same step as a +8)
            life = info.get('life', prev_life)
            if info.get('loop_timeout'):
                event, event_ttl = 'LOOP %+.0f' % reward, 60
            elif info.get('wrong_exit'):
                event, event_ttl = 'OFF-ROUTE %+.0f' % reward, 60
            elif info.get('timeout'):
                event, event_ttl = 'STUCK -> TIME-UP', 60
            elif life == 255 and prev_life is not None and prev_life != 255:
                event, event_ttl = 'GAME OVER %+.0f' % reward, 60
            elif prev_life is not None and life < prev_life:
                event, event_ttl = 'DEATH %+.0f' % reward, 60
            elif info.get('flag_get') or info.get('victory'):
                event, event_ttl = 'CLEAR %+.0f' % reward, 60
            prev_life = life
            stat = (info.get('world', 1), info.get('stage', 1),
                    info.get('x_pos', 0), life_reward, reward,
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
        cause = ('victory' if info.get('victory') else 'loop' if info.get('loop_timeout')
                 else 'unpaid timeout' if info.get('timeout') else 'wrong exit'
                 if info.get('wrong_exit') else 'game over' if info.get('life') == 255
                 else 'level cleared' if (gp0 is not None and info.get('game_progress') != gp0)
                 else 'step cap')
        print(f'  [Video] clip {info.get("world", "?")}-{info.get("stage", "?")}: '
              f'{len(frames)} frames, ended by {cause} at x={info.get("x_pos", 0)}')
        font = ImageFont.load_default()
        pil_frames = [self.draw_strip(f, epoch_num, s, font)
                      for s, f in zip(step_stats, frames)]
        if tr_env is not None and tr_rows:
            self._dump_video_trace(epoch_num, info, tr_state, tr_acts, tr_rows,
                                   list(tr_env.last_terms.keys()), start_lvl=tr_start_lvl,
                                   raw=int(bool(getattr(tr_env, '_raw_steps', False))))
        return frames, pil_frames, step_stats, info, total_reward, per_step_frames

    def _dump_video_trace(self, epoch_num, info, state, acts, rows, term_names, start_lvl=None, raw=1):
        """<run>/eval_traces/epoch_N/video_<start level>.csv (+ .npz with the
        start state and actions for tools/play.py --replay; `raw` records
        whether the clip was stepped hack-free, which the replay must match)."""
        try:
            run_dir = self._run_dir()
            if run_dir is None:
                return
            out = os.path.join(run_dir, 'eval_traces', f'epoch_{epoch_num}')
            os.makedirs(out, exist_ok=True)
            # named by the level the clip STARTED in (the end level was
            # misleading: the 8-1 clip that cleared into 8-2 was 'video_8-2')
            lvl = start_lvl or '%s-%s' % (info.get('world', '?'), info.get('stage', '?'))
            tag = f'video_{lvl}_{len(os.listdir(out)):02d}'
            with open(os.path.join(out, tag + '.csv'), 'w') as f:
                f.write(','.join(['step', 'action', 'x', 'ypix', 'life', 'reward']
                                 + term_names + ['page_reset', 'timeout', 'died',
                                                 'transition', 'level_clear', 'done']) + '\n')
                for r in rows:
                    f.write(','.join(str(v) for v in r) + '\n')
            np.savez_compressed(os.path.join(out, tag + '.npz'),
                                state=np.frombuffer(state, dtype=np.uint8),
                                actions=np.array(acts, dtype=np.int16),
                                term_names=np.array(term_names), epoch=epoch_num,
                                raw=int(raw))
        except Exception as e:
            print(f'  [Video] trace dump failed: {e}')

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

    @staticmethod
    def _sample_actions(logits, gen):
        """Actions drawn from the policy with a local generator: reproducible
        for a fixed seed, and independent of the process-wide torch RNG."""
        probs = torch.softmax(logits, -1)
        return torch.multinomial(probs, 1, generator=gen).squeeze(1).numpy()

    def _eval_env_config(self):
        """Training env config stripped of noise, restarts and recorder keys."""
        ec = dict(self.algo.env_config or {})
        route = list(ec.get('route_levels') or ec.get('random_stages') or [])
        for k in ('name', 'action_type', 'archive_path', 'video_levels',
                  'video_level_steps', 'backend'):
            ec.pop(k, None)
        ec.update(sticky_actions=0.0, explore_eps=0.0, self_restart_prob=0.0,
                  explore_episode_prob=0.0, reset_noops=0, n_threads=4,
                  dense_infos=False, route_levels=route, full_game=True)
        return ec, route

    def _level_eval(self, model, epoch_num, levels=None, n=None, max_steps=None,
                    seed=0):
        """Per-level door evaluation of the SAMPLED policy: n single-life
        episodes from each level's own door state, no noise, training-style
        stepping, fixed seed (reproducible, and the episodes are not clones
        of each other). Writes eval/level_clear/<lvl> (the clear rate) and the
        max-x / timeout / wrong-exit / death rates, an index of every episode
        and the best + worst replayable trace per level.

        Why not argmax: PPO trains the sampled policy; argmax is a different
        policy with fixed points (a frame that maps to 'run right' into a
        pipe repeats forever) and one trajectory per level, so its clear
        flags flipped 0/1 between epochs while the sampled clear rate of the
        same checkpoint was 0.44-0.66 (2026-09-11 review)."""
        from mario_native_vecenv import MarioNativeVecEnv
        n = int(n or self.eval_episodes); max_steps = int(max_steps or self.eval_level_steps)
        ec, route = self._eval_env_config()
        levels = list(levels or route)
        out, dump = {}, []
        for li, lvl in enumerate(levels):
            gen = torch.Generator().manual_seed(int(seed) * 1000 + li)
            gp0 = (int(lvl[0]) - 1) * 4 + int(lvl[2]) - 1
            env = MarioNativeVecEnv('leval', n, **dict(ec, random_stages=[lvl], episode_life=True,
                                                       seed=int(seed) * 1000 + li))
            try:
                obs = env.reset()
                starts = []
                for i in range(n):
                    env.lib.benv_save(env.env, i, env._sbuf)
                    starts.append(bytes(env._sbuf.raw))
                acts = [[] for _ in range(n)]; terms = [[] for _ in range(n)]
                xs = [[] for _ in range(n)]
                fin, maxx, clear = {}, np.zeros(n, dtype=np.int64), np.zeros(n, dtype=bool)
                term_names = None
                for step in range(max_steps):
                    with torch.no_grad():
                        lg = model({'obs': torch.from_numpy(obs).float(), 'is_train': False})['logits']
                    a = self._sample_actions(lg, gen)
                    obs, r, d, infos = env.step(a)
                    lt = env.last_terms; term_names = list(lt.keys())
                    for i in range(n):
                        if i in fin:
                            continue
                        acts[i].append(int(a[i])); terms[i].append([float(v[i]) for v in lt.values()])
                        xs[i].append(int(env.last_signals.x[i]))
                        if d[i]:
                            inf = infos[i] if isinstance(infos, list) else {}
                            maxx[i] = int(inf.get('max_x_pos', 0))
                            clear[i] = inf.get('stages_cleared', 0) > 0 or bool(inf.get('victory', False))
                            fin[i] = ('clear' if clear[i] else 'wrong_exit' if inf.get('wrong_exit')
                                      else 'timeout' if inf.get('timeout') else 'death')
                        elif int(env.progress[i]) > gp0:
                            # the level was left by its route exit: the episode
                            # keeps playing the next level (a clear is not a
                            # terminal), but for THIS level's rate it is over.
                            # Counting only at done marked every such episode
                            # 'running' when it hit the step cap in the next level.
                            maxx[i] = int(env.max_x[i]); clear[i] = True; fin[i] = 'clear'
                        else:
                            maxx[i] = int(env.max_x[i])
                    if len(fin) == n:
                        break
                for i in range(n):
                    fin.setdefault(i, 'running')
            finally:
                env.close()
            ends = {k: sum(1 for v in fin.values() if v == k) / n
                    for k in ('clear', 'death', 'timeout', 'wrong_exit', 'running')}
            out[lvl] = {'clear': float(clear.mean()), 'max_x': [int(v) for v in maxx], 'ends': ends}
            self.writer.add_scalar(f'eval/level_clear/{lvl}', float(clear.mean()), epoch_num)
            self.writer.add_scalar(f'eval/level_max_x_mean/{lvl}', float(maxx.mean()), epoch_num)
            self.writer.add_scalar(f'eval/level_timeout_rate/{lvl}', ends['timeout'], epoch_num)
            self.writer.add_scalar(f'eval/level_wrong_exit_rate/{lvl}', ends['wrong_exit'], epoch_num)
            self.writer.add_scalar(f'eval/level_death_rate/{lvl}', ends['death'], epoch_num)
            order = np.argsort(-maxx)
            for i in range(n):
                dump.append((lvl, i, fin[i], int(maxx[i]), acts[i], terms[i], xs[i], starts[i],
                             i in (order[0], order[-1]), term_names))
            print(f'  [Eval] {lvl} x{n} sampled: clear {clear.mean():.2f} max_x mean '
                  f'{maxx.mean():.0f} | ' + ' '.join(f'{k} {v:.2f}' for k, v in ends.items()))
        if levels:
            self.writer.add_scalar('eval/level_clear_mean',
                                   float(np.mean([out[l]['clear'] for l in levels])), epoch_num)
        self._dump_level_traces(epoch_num, dump)
        return out

    def _dump_level_traces(self, epoch_num, dump):
        """index.csv of every sampled door episode + best / worst replayable
        .npz per level under <run>/eval_traces/epoch_<N>/ (training-style
        stepping: raw=0, which tools/play.py --replay honours)."""
        try:
            run_dir = self._run_dir()
            if run_dir is None:
                return
            out = os.path.join(run_dir, 'eval_traces', f'epoch_{epoch_num}')
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, 'index.csv'), 'w') as f:
                f.write('level,episode,end,max_x,steps,total_reward,file\n')
                for lvl, i, end, mx, acts, terms, xs, start, keep, names in dump:
                    tot = float(np.sum(terms)) if terms else 0.0
                    fn = ''
                    if keep:
                        fn = f'ep_{lvl}_{i:02d}_{end}_x{mx}.npz'
                        np.savez_compressed(
                            os.path.join(out, fn), state=np.frombuffer(start, dtype=np.uint8),
                            actions=np.array(acts, dtype=np.int16),
                            terms=np.array(terms, dtype=np.float32),
                            term_names=np.array(names or []), x=np.array(xs),
                            level=str(lvl), epoch=epoch_num, raw=0)
                    f.write(f'{lvl},{i},{end},{mx},{len(acts)},{tot:.1f},{fn}\n')
        except Exception as e:
            print(f'  [Eval] trace dump failed: {e}')

    def _run_dir(self):
        try:
            return os.path.dirname(os.path.dirname(
                self.writer.file_writer.event_writer._ev_writer._file_name))
        except AttributeError:
            logdir = getattr(self.writer, 'logdir', None) or \
                getattr(self.writer, 'log_dir', None)
            if not logdir:
                return None
            return os.path.dirname(os.path.normpath(logdir))

    @staticmethod
    def route_progress(gp, route_gps):
        """Route-aware level index: the last ON-route level at or below gp.
        The env's progress counter also records the level of a wrong exit
        (4-2 flag -> 4-3 = 14), which is a failure at 4-2, not progress."""
        gp = int(gp)
        if not route_gps or gp in route_gps:
            return gp
        below = [g for g in route_gps if g < gp]
        return max(below) if below else gp

    def _sequential_eval(self, model, epoch_num, n=None, max_steps=None, seed=0):
        """The real objective as a rate: n seeded SAMPLED-policy games from
        the first level with 3 lives (training-style stepping, no noise).
        Writes eval/game_progress_sampled_mean / _max (last ON-route level
        index reached, 0-31), eval/victory_rate_sampled and
        eval/off_route_exit_rate_sampled (games that ended by a wrong exit)."""
        from mario_native_vecenv import MarioNativeVecEnv
        n = int(n or self.eval_seq_episodes); max_steps = int(max_steps or self.eval_seq_steps)
        ec, route = self._eval_env_config()
        gen = torch.Generator().manual_seed(int(seed) * 7919 + 1)
        route_gps = {(int(l[0]) - 1) * 4 + int(l[2]) - 1 for l in route}
        env = MarioNativeVecEnv('seval', n, **dict(ec, random_stages=None, episode_life=False,
                                                   seed=int(seed) * 7919 + 1))
        try:
            obs = env.reset()
            done_m = np.zeros(n, dtype=bool); gp = np.zeros(n, dtype=np.int64)
            vic = np.zeros(n, dtype=bool); off = np.zeros(n, dtype=bool)
            for step in range(max_steps):
                with torch.no_grad():
                    lg = model({'obs': torch.from_numpy(obs).float(), 'is_train': False})['logits']
                a = self._sample_actions(lg, gen)
                obs, r, d, infos = env.step(a)
                gp = np.where(done_m, gp, np.maximum(gp, env.progress))
                for i in np.nonzero(d & ~done_m)[0]:
                    inf = infos[i] if isinstance(infos, list) else {}
                    done_m[i] = True
                    vic[i] = bool(inf.get('victory', False))
                    off[i] = bool(inf.get('wrong_exit', False))
                    gp[i] = max(gp[i], int(inf.get('game_progress', 0)))
                if done_m.all():
                    break
        finally:
            env.close()
        gp = np.array([self.route_progress(g, route_gps) for g in gp], dtype=np.int64)
        res = {'progress_mean': float(gp.mean()), 'progress_max': int(gp.max()),
               'victory': float(vic.mean()), 'off_route': float(off.mean())}
        self.writer.add_scalar('eval/game_progress_sampled_mean', res['progress_mean'], epoch_num)
        self.writer.add_scalar('eval/game_progress_sampled_max', res['progress_max'], epoch_num)
        self.writer.add_scalar('eval/victory_rate_sampled', res['victory'], epoch_num)
        self.writer.add_scalar('eval/off_route_exit_rate_sampled', res['off_route'], epoch_num)
        lv = lambda g: '%d-%d' % (g // 4 + 1, g % 4 + 1)
        print(f'  [Eval] sequential x{n} sampled, 3 lives: level reached mean '
              f'{res["progress_mean"]:.1f} max {lv(res["progress_max"])} victory {res["victory"]:.2f} '
              f'off-route exits {res["off_route"]:.2f}')
        return res

    def _record_video(self, epoch_num, model):
        """Record gameplay: PIL GIF to TensorBoard + MP4 to disk.

        Runs on a background thread with a CPU-only copy of the model, so
        training continues undisturbed and no GPU is touched from the thread.
        """
        try:
            import imageio
            import tempfile
            from PIL import Image, ImageDraw, ImageFont
            # sampled-policy evaluations (the numbers to trust); the clips
            # below play the sampled policy too, seeded per epoch and level
            # (reproducible); their scalars carry a _clip suffix (one episode)
            try:
                self._level_eval(model, epoch_num, seed=epoch_num)
            except Exception as e:
                print(f'  [Eval] level eval failed: {e}')
            try:
                self._sequential_eval(model, epoch_num, seed=epoch_num)
            except Exception as e:
                print(f'  [Eval] sequential eval failed: {e}')
            try:
                from tensorboardX.proto.summary_pb2 import Summary
            except ImportError:
                from tensorboard.compat.proto.summary_pb2 import Summary

            # main clip: sequential game from the eval start (1-1 by default)
            (frames, pil_frames, step_stats, info, total_reward,
             per_step) = self._play_clip(model, self._make_eval_env(),
                                         self.video_max_steps, epoch_num,
                                         seed=epoch_num * 100)

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
                    for li, lvl in enumerate(levels):
                        env_l = self._make_eval_env(random_stages=[lvl],
                                                    full_game=True)
                        fr, pf, st, inf_l, rew_l, ps = self._play_clip(
                            model, env_l, lvl_steps, epoch_num,
                            stop_on_level_change=True,
                            seed=epoch_num * 100 + li + 1)
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
                        self.writer.add_scalar(f'eval/level_max_x_clip/{lvl}',
                                               max(s[2] for s in st),
                                               epoch_num)
                        # cleared = left the level by its route exit
                        lvl_gp = (int(lvl[0]) - 1) * 4 + int(lvl[2]) - 1
                        cleared = int(inf_l.get('game_progress', lvl_gp) > lvl_gp
                                      and not inf_l.get('wrong_exit', False))
                        self.writer.add_scalar(f'eval/level_clear_clip/{lvl}',
                                               cleared, epoch_num)
                        sizes.append(len(gb) // 1024)
                    self.writer.flush()
                    print(f'  [Video] Epoch {epoch_num}: {len(sizes)} level '
                          f'clips, GIF KB {sizes}')

                x_pos = info.get('x_pos', 0)
                world = info.get('world', 1)
                stage = info.get('stage', 1)
                # Eval frontier scalars of the sequential CLIP from 1-1 (one seeded game)
                # (eval/game_progress_sampled_* hold the sampled-policy rate)
                self.writer.add_scalar('eval/game_progress',
                                       info.get('game_progress', 0), epoch_num)
                # route-aware: the level index only counts while on the
                # configured route; an off-route exit (1-2 flag -> 1-3,
                # warp pipe 2 -> 2-1) is flagged instead of counted
                route = (self.algo.env_config or {}).get('random_stages') or []
                rgp = {(int(l[0]) - 1) * 4 + int(l[2]) - 1 for l in route}
                gp_now = int(info.get('game_progress', 0))
                self.writer.add_scalar('eval/route_progress',
                                       gp_now if (not rgp or gp_now in rgp) else -1, epoch_num)
                self.writer.add_scalar('eval/off_route_exit',
                                       int(bool(rgp) and gp_now not in rgp), epoch_num)
                self.writer.add_scalar(
                    'eval/max_x', max(s[2] for s in step_stats), epoch_num)
                print(f'  [Video] Epoch {epoch_num}: reward={total_reward:.0f}, '
                      f'world={world}-{stage}, x_pos={x_pos}, '
                      f'gif={len(gif_bytes)/1024:.0f}KB, mp4={mp4_path}')

        except Exception as e:
            import traceback
            print(f'  [Video] Recording failed: {e}')
            traceback.print_exc()
