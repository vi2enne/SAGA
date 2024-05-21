- Point Maze
```
./scripts/saga_point_maze.sh ${reward_shaping} ${timesteps} ${gpu} ${seed}
./scripts/saga_point_maze.sh dense 5e5 0 2
./scripts/saga_point_maze.sh sparse 5e5 0 2
```

- Ant Maze (U-shape)
```
./scripts/saga_ant_maze_u.sh ${reward_shaping} ${timesteps} ${gpu} ${seed}
./scripts/saga_ant_maze_u.sh dense 10e5 0 2
./scripts/saga_ant_maze_u.sh sparse 10e5 0 2
```

- Ant Maze (W-shape)
```
./scripts/saga_ant_maze_w.sh ${reward_shaping} ${timesteps} ${gpu} ${seed}
./scripts/saga_ant_maze_w.sh dense 10e5 0 2
./scripts/saga_ant_maze_w.sh sparse 10e5 0 2
```

- Reacher & Pusher
```
./scripts/saga_fetch.sh ${env} ${timesteps} ${gpu} ${seed}
./scripts/saga_fetch.sh Reacher3D-v0 5e5 0 2
./scripts/saga_fetch.sh Pusher-v0 10e5 0 2
```

- Stochastic Ant Maze (U-shape)
```
./scripts/saga_ant_maze_u_stoch.sh ${reward_shaping} ${timesteps} ${gpu} ${seed}
./scripts/saga_ant_maze_u_stoch.sh dense 10e5 0 2
./scripts/saga_ant_maze_u_stoch.sh sparse 10e5 0 2
```

