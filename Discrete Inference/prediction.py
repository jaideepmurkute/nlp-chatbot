
from CFG import Config
from discrete_dataset_inference import DiscreteInference

if __name__ == "__main__":
    cfg = Config().config
    di = DiscreteInference(cfg)
    di.run_inference()