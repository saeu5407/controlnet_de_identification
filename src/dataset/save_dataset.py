from datasets import load_dataset

dataset = load_dataset("celebhq_landmark.py",
                       data_dir="/home/analysis02/tugboat-data/projectdata/celebahq")

dataset.push_to_hub("saeu5407/celebahq_landmark4controlnet")