import json

categories = ['Employment', 'Housing', 'Food', 'Financial', 'Transportation', 'Childcare', 'Permanency', 'Substance', 'Home', 'Community']

def parse_predictions(predictions):
    # json object or "NoSocialNeedsFoundLabel" or list
    assert isinstance(predictions, (dict, list)) or predictions == "NoSocialNeedsFoundLabel"

    if isinstance(predictions, dict):
        ret = {}

        for cat in categories:
            if cat in predictions:
                if predictions[cat] == "None":
                    ret[cat] = 0
                else:
                    ret[cat] = len(predictions[cat])
            else:
                ret[cat] = 0
        return ret
    elif isinstance(predictions, list):
        ret = []
        for p in predictions:
            if p == "NoSocialNeedsFoundLabel":
                ret.append("NoSocialNeedsFoundLabel")
            else:
                assert isinstance(p, dict)
                ret.append(parse_predictions(p))
        return ret
    else:
        raise ValueError(f"Invalid predictions: {predictions}")



def evaluate(predictions, labels):
    pass


if __name__ == "__main__":
    predictions = parse_predictions(json.load(open("output/test/llama_output.json")))
    print(predictions)