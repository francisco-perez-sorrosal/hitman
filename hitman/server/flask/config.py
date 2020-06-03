from environs import Env, EnvError

env = Env()
env.read_env()

# App Setup
DUMMY_REQ_PROC_TIME_SECS = env.float("DUMMY_REQ_PROC_TIME_SECS", 1.0)
print("Proc time {}".format(DUMMY_REQ_PROC_TIME_SECS))

PYTORCH_MODEL_PATH = env.str("PYTORCH_MODEL_PATH", "/export/sieve/oic/checkpoint")
print("PYTORCH_MODEL_PATH {}".format(PYTORCH_MODEL_PATH))