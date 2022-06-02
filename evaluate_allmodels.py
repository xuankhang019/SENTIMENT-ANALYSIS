import string
import csv
from sklearn.model_selection import train_test_split

from tensorflow.python.keras.models import Sequential
import os
from module.evaluate import cal_sentiment_prf
from module.model.dt import PolarityDTModel
from module.model.lr import PolarityLRModel
from module.model.svm import PolaritySVMModel

from module.preprocess import preprocess, load_data

input_data = [
    'data/raw_data/mebe_tiki.csv',
]
LABEL = {
    'mebe_tiki.csv': 'aspect0,aspect1,aspect2,aspect3,aspect4,aspect5'
    # 'data_train/mebe_shopee.csv': 'giá,dịch_vụ,an_toàn,chất_lượng,ship,chính_hãng',
}

NUM_OF_ASPECTS = 6
NUM_OF_MODEL = 3

modelNames = ["Decision Tree", "Logistic Regression", "Support Vector Machine"]
models = [PolarityDTModel(), PolarityLRModel(), PolaritySVMModel()]

if __name__ == '__main__':
    for f in input_data:
        name = f.split('/')
        name = name[2].split('.')
        name = name[0]

        abs_file_path = "data/raw_data/mebe_tiki.csv"

        print(name)
        print('Running {}...'.format(f))
        tp = []
        fp = []
        fn = []

        for model_i_th in range(NUM_OF_MODEL):
            model = models[model_i_th]
            for aspectId in range(NUM_OF_ASPECTS):
                inputs, outputs = load_data(abs_file_path, aspectId)
                inputs = preprocess(inputs)

                # print(inputs[0].text)
                Sequential().compile()

                # five_folds_cross_validation(inputs, outputs, model, aspectId=aspectId)
                X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=14)
                model.train(X_train, y_train, aspectId)
                predicts = model.predict(X_test, aspectId)
                # print(predicts[0].getall())
                _tp, _fp, _fn, p, r, f1 = model.evaluate(y_test, predicts)
                print(p, r, f1)
                tp.append(_tp)
                fp.append(_fp)
                fn.append(_fn)
                modelName = str(model)
                modelName = modelName[(modelName.index(".") + 1):modelName.index(" ")]

            cal_sentiment_prf(tp, fp, fn, NUM_OF_ASPECTS, verbal=True, modelName=modelNames[model_i_th],
                              fisrtModel=(True if model_i_th == 0 else False))