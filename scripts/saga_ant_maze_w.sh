REWARD_SHAPING=$1
TIMESTEPS=$2
GPU=$3
SEED=$4

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
--env_name "AntMazeW-v2" \
--reward_shaping ${REWARD_SHAPING} \
--algo saga \
--version "${REWARD_SHAPING}" \
--seed ${SEED} \
--max_timesteps ${TIMESTEPS} \
--landmark_sampling fps \
--n_landmark_coverage 60 \
--use_novelty_landmark \
--novelty_algo rnd \
--n_landmark_novelty 60 \
--seed ${SEED}
