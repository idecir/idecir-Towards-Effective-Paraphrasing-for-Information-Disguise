import numpy as np


def fetch_use_embedding(model, input_sentence: str) -> np.ndarray:
    # returns USE embedding (512 dimensions)
    return model([input_sentence]).numpy()[0]


if __name__ == "__main__":
    import tensorflow_hub as hub
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    # module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    use_model = hub.load(module_url)

    s1 = "It is a wonderful morning"
    s2 = "It is a bad morning"
    s3 = "It is a pleasant morning"
    e1 = fetch_use_embedding(use_model, s1)
    e2 = fetch_use_embedding(use_model, s2)
    e3 = fetch_use_embedding(use_model, s3)

    print("Sim between s1 and s2 is ", np.inner(e1, e2))  # 0.76
    print("Sim between s1 and s3 is ", np.inner(e1, e3))  # 0.89
    print("Sim between s2 and s3 is ", np.inner(e3, e2))  # 0.75
