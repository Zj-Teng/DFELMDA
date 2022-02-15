import numpy as np
from keras import layers, models
from sklearn.ensemble import RandomForestClassifier
from utils import prepare_data, preprocess_labels


def miRNA_auto_encoder(x_train):
    encoding_dim = 64
    input_img = layers.Input(shape=(990,))
    # 建立神经网络
    # 编码层
    encoded = layers.Dense(350, activation='relu')(input_img)
    encoded = layers.Dense(250, activation='relu')(encoded)
    encoded = layers.Dense(100, activation='relu')(encoded)
    miRNA_encoder_output = layers.Dense(encoding_dim)(encoded)
    # 解码层
    decoded = layers.Dense(100, activation='relu')(miRNA_encoder_output)
    decoded = layers.Dense(250, activation='relu')(decoded)
    decoded = layers.Dense(350, activation='relu')(decoded)
    decoded = layers.Dense(990, activation='tanh')(decoded)
    # 构建自编码模型
    autoencoder = models.Model(input=input_img, output=decoded)
    encoder = models.Model(input=input_img, output=miRNA_encoder_output)
    # 激活模型
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(x_train, x_train, epochs=20, batch_size=100, shuffle=True)
    miRNA_encoded_imgs = encoder.predict(x_train)
    return miRNA_encoder_output, miRNA_encoded_imgs


def disease_auto_encoder(y_train):
    encoding_dim = 64
    input_img = layers.Input(shape=(766,))
    # 建立神经网络
    # 编码层
    encoded = layers.Dense(350, activation='relu')(input_img)
    encoded = layers.Dense(250, activation='relu')(encoded)
    encoded = layers.Dense(100, activation='relu')(encoded)
    disease_encoder_output = layers.Dense(encoding_dim)(encoded)
    # 解码层
    decoded = layers.Dense(100, activation='relu')(disease_encoder_output)
    decoded = layers.Dense(250, activation='relu')(decoded)
    decoded = layers.Dense(350, activation='relu')(decoded)
    decoded = layers.Dense(766, activation='tanh')(decoded)
    # 构建自编码模型
    autoencoder = models.Model(input=input_img, output=decoded)
    encoder = models.Model(input=input_img, output=disease_encoder_output)
    # 激活模型
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(y_train, y_train, epochs=20, batch_size=100, shuffle=True)
    disease_encoded_imgs = encoder.predict(y_train)
    return disease_encoder_output, disease_encoded_imgs

def AEMDA():
    mtrain, dtrain, label = prepare_data()
    m, encoder = preprocess_labels(label)
    nm = np.arange(len(m))
    m = m[nm]

    encoder, m_data1 = miRNA_auto_encoder(mtrain)
    encoder, d_data1 = disease_auto_encoder(dtrain)

    num_cross = 5

    probaresult = []
    ae_y_pred_probresult = []

    for fold in range(num_cross):
        train_m = np.array([x for i, x in enumerate(m_data1) if i % num_cross != fold])
        test_m = np.array([x for i, x in enumerate(m_data1) if i % num_cross == fold])
        train_d = np.array([x for i, x in enumerate(d_data1) if i % num_cross != fold])
        test_d = np.array([x for i, x in enumerate(d_data1) if i % num_cross == fold])
        train_label = np.array([x for i, x in enumerate(m) if i % num_cross != fold])

        train_label_new = []

        for val in train_label:
            if val[0] == 1:
                train_label_new.append(0)
            else:
                train_label_new.append(1)

        prefilter_mtrain = train_m
        prefilter_mtest = test_m
        prefilter_dtrain = train_d
        prefilter_dtest = test_d


        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(prefilter_mtrain, train_label_new)
        mae_y_pred_prob = clf.predict_proba(prefilter_mtest)[:, 1]
        mproba = transfer_label_from_prob(mae_y_pred_prob)

        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(prefilter_dtrain, train_label_new)
        dae_y_pred_prob = clf.predict_proba(prefilter_dtest)[:, 1]
        dproba = transfer_label_from_prob(dae_y_pred_prob)

        mproba = np.array(mproba)
        dproba = np.array(dproba)

        proba = (mproba + dproba)/2
        ae_y_pred_prob = (mae_y_pred_prob + dae_y_pred_prob)/2

        probaresult.extend(proba)
        ae_y_pred_probresult.extend(ae_y_pred_prob)

    return probaresult, ae_y_pred_probresult, m

def transfer_label_from_prob(proba):
    label = [1 if val >= 0.5 else 0 for val in proba]
    return label