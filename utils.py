from keras.callbacks import TensorBoard, ModelCheckpoint
from datetime import datetime
import os


def create_logger(env_name: str, **kwargs):
    date = datetime.today().strftime('%Y-%m-%d_%H%M')

    log_dir = 'logs/%s/%s' % (env_name, date)
    os.makedirs(log_dir)

    return TensorBoard(log_dir=log_dir, **kwargs)


def create_model_checkpoint(env_name: str, monitor, mode, **kwargs):
    date = datetime.today().strftime('%Y-%m-%d_%H%M')
    checkpoint_dir = 'models/checkpoints/%s/%s' % (env_name, date)
    os.makedirs(checkpoint_dir)
    return checkpoint_dir, ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'weights_{epoch:02d}-{episode_reward:.0f}.h5'),
        monitor=monitor, mode=mode, **kwargs)
