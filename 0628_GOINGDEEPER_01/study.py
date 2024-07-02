#%% 라이브러리 로드
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds

import pandas as pd

# Tensorflow가 활용할 GPU가 장착되어 있는지 확인해 봅니다.
tf.config.list_physical_devices('GPU')


#%% 상수
BATCH_SIZE = 32  # 256
EPOCHS = 10  # 15
LR = 0.1
SEED = 42
OPT_DECAY = 0.0001
MOMENTUM = 0.9
L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

ACTIVATION = 'sigmoid'
NUM_CLASSES = 1

#%% 데이터셋 로드
(ds_train, ds_test), ds_info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True,
    shuffle_files=True,
    with_info=True,
)

# Tensorflow 데이터셋을 로드하면 꼭 feature 정보를 확인해 보세요. 
print(ds_info.features)
# 데이터의 개수도 확인해 봅시다. 
print(tf.data.experimental.cardinality(ds_train))
print(tf.data.experimental.cardinality(ds_test))
# 클래스 수
print(ds_info.features["label"].num_classes)
# 클래스 이름
print(ds_info.features["label"].names)
# 예시 이미지 (주피터 노트북 용)
fig = tfds.show_examples(ds_train, ds_info)

# 
for input in ds_train.take(1):
    image, label = input
    print(image.shape)
    print(label.shape)

for input in ds_test.take(1):
    image, label = input
    print(image.shape)
    print(label.shape)

#%% 전처리 함수
def normalize_and_resize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.image.resize(image, (224,224))
    image = tf.cast(image, tf.float32) / 255.
    label = tf.one_hot(label, depth=2)  # 원-핫 인코딩 추가
    return image, label

def apply_normalize_on_dataset(ds, is_test=False, batch_size=16):
    ds = ds.map(
        normalize_and_resize_img, 
        num_parallel_calls=1
    )
    ds = ds.batch(batch_size)
    if not is_test:
        ds = ds.repeat()
        ds = ds.shuffle(200)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

#%% 데이터셋 전처리
ds_train = apply_normalize_on_dataset(ds_train, batch_size=BATCH_SIZE)
ds_test = apply_normalize_on_dataset(ds_test, batch_size=BATCH_SIZE)

for input in ds_train.take(1):
    image, label = input
    print(image.shape)
    print(label.shape)

for input in ds_test.take(1):
    image, label = input
    print(image.shape)
    print(label.shape)

#%% 모델 함수 
def conv_block(input_layer, 
                channel, 
                kernel_size, 
                strides=1, 
                activation='relu',
                # l2_weight_decay=1e-4,
                # batch_norm_decay=0.9,
                # batch_norm_epsilon=1e-5,
                ):
    
    x = keras.layers.Conv2D(
        filters=channel,
        kernel_size=kernel_size,
        kernel_initializer='he_normal',
        # kernel_regularizer=keras.regularizers.l2(l2_weight_decay),
        padding='same',
        strides=strides,
        # use_bias=False,
    )(input_layer)
    
    x = keras.layers.BatchNormalization()(x)
    
    if activation:
        x = keras.layers.Activation(activation)(x)
    
    return x

def build_resnet34_block(input_layer, 
                        cnn_count=3, 
                        channel=64, 
                        block_num=0,
                        ):

    x = input_layer
    
    # 첫 번째 conv_layer 에서 strides=2 설정해서 너비와 높이를 줄임
    # => 필터의 개수(=채널)가 증가할 때마다 너비와 높이를 줄임
    for i in range(cnn_count):
        if block_num > 0 and i == 0:
            shortcut = conv_block(x, channel, (1,1), strides=2, activation=None)
            x = conv_block(x, channel, (3,3), strides=2)
            x = conv_block(x, channel, (3,3), activation=None)
        else:
            shortcut = x
            x = conv_block(x, channel, (3,3))
            x = conv_block(x, channel, (3,3), activation=None)
        
        x = keras.layers.Add()([x, shortcut])
        x = keras.layers.Activation('relu')(x)

    return x

def build_resnet50_block(input_layer, 
                        cnn_count=3, 
                        channel=64, 
                        block_num=0,
                        ):

    x = input_layer
    
    # 첫 번째 conv_layer 에서 strides=2 설정해서 너비와 높이를 줄임
    # => 필터의 개수(=채널)가 증가할 때마다 너비와 높이를 줄임
    for i in range(cnn_count):
        if i == 0:
            shortcut = conv_block(x, channel*4, (1,1), strides=2, activation=None)
            x = conv_block(x, channel, (1,1), strides=2)
            x = conv_block(x, channel, (3,3))
            x = conv_block(x, channel*4, (1,1), activation=None)
        else:
            shortcut = x
            x = conv_block(x, channel, (1,1))
            x = conv_block(x, channel, (3,3))
            x = conv_block(x, channel*4, (1,1), activation=None)
        
        x = keras.layers.Add()([x, shortcut])
        x = keras.layers.Activation('relu')(x)

    return x

def build_plainnet34_block(input_layer, 
                        cnn_count=3, 
                        channel=64, 
                        block_num=0,
                        ):

    x = input_layer
    
    for i in range(cnn_count):
        if block_num > 0 and i == 0:
            x = conv_block(x, channel, (3,3), strides=2)
            x = conv_block(x, channel, (3,3))
        else:
            x = conv_block(x, channel, (3,3))
            x = conv_block(x, channel, (3,3))

    return x

def build_plainnet50_block(input_layer, 
                        cnn_count=3, 
                        channel=64, 
                        block_num=0,
                        ):

    x = input_layer
    
    for i in range(cnn_count):
        if block_num > 0 and i == 0:
            x = conv_block(x, channel, (1,1), strides=2)
            x = conv_block(x, channel, (3,3))
            x = conv_block(x, channel*4, (1,1))
        else:
            x = conv_block(x, channel, (1,1))
            x = conv_block(x, channel, (3,3))
            x = conv_block(x, channel*4, (1,1))

    return x


def build_net(input_shape=(32,32,3),
                cnn_count_list=[3,4,6,3],
                channel_list=[64,128,256,512],
                num_classes=10, 
                activation='softmax',
                name='ResNet_50'):
    
    # 모델 생성 전, config list들이 같은 길이인지 확인
    assert len(cnn_count_list) == len(channel_list)

    # 모델 설정
    model_func = globals()[f'build_{name}_block']
    
    # input layer 생성
    input_layer = keras.layers.Input(shape=input_shape)
    
    # first layer 설정
    x = conv_block(input_layer, 64, (7,7), strides=2)
    x = keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(x)
    
    # config list들의 길이만큼 반복해서
    # - is_ResNet=True 일 경우, Residual block 블록을 생성합니다.
    # - is_ResNet=False 일 경우, Plain block 블록을 생성합니다.
    for block_num, (cnn_count, channel) in enumerate(zip(cnn_count_list, channel_list)):
        x = model_func(x, 
                        cnn_count=cnn_count, 
                        channel=channel, 
                        block_num=block_num, 
                        )
        
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(num_classes, 
                        activation=activation, 
                        kernel_initializer='he_normal'
                        )(x)
    
    model = keras.Model(inputs=input_layer, outputs=x, name=name)
    
    return model

#%% 모델 생성
resnet_34 = build_net(input_shape=(224,224,3),
                    cnn_count_list=[3,4,6,3],
                    channel_list=[64,128,256,512],
                    num_classes=2, 
                    activation='softmax',
                    name='resnet34')

# resnet_34.summary()

resnet_50 = build_net(input_shape=(224,224,3),
                    cnn_count_list=[3,4,6,3],
                    channel_list=[64,128,256,512],
                    num_classes=2, 
                    activation='softmax',
                    name='resnet50')

# resnet_50.summary()

plain_34 = build_net(input_shape=(224,224,3),
                    cnn_count_list=[3,4,6,3],
                    channel_list=[64,128,256,512],
                    num_classes=2, 
                    activation='softmax',
                    name='plainnet34')

# plain_34.summary()

plain_50 = build_net(input_shape=(224,224,3),
                    cnn_count_list=[3,4,6,3],
                    channel_list=[64,128,256,512],
                    num_classes=2, 
                    activation='softmax',
                    name='plainnet50')

# plain_50.summary()

#%% 학습
def comile_and_fit(model, ds_train, ds_test, ds_info, learning_rate=0.001,
                   momentum= 0.9, opt_decay=0.0001, 
                   batch_size=128, epochs=15):
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            # momentum=momentum,
            # decay=opt_decay,
            clipnorm=1.
        ),
        metrics=['accuracy'],
    )

    history = model.fit(
        ds_train,
        steps_per_epoch=int(ds_info.splits['train[:80%]'].num_examples/batch_size),
        validation_steps=int(ds_info.splits['train[80%:]'].num_examples/batch_size),
        epochs=epochs,
        validation_data=ds_test,
        verbose=1,
        # use_multiprocessing=True,
    )

    return history

history_resnet_34 = comile_and_fit(resnet_34, ds_train, ds_test, ds_info, LR, 
                                #    MOMENTUM, OPT_DECAY, 
                                   BATCH_SIZE, EPOCHS)

history_plain_34 = comile_and_fit(plain_34, ds_train, ds_test, ds_info, LR,
                                #   MOMENTUM, OPT_DECAY,
                                   BATCH_SIZE, EPOCHS)

history_resnet_50 = comile_and_fit(resnet_50, ds_train, ds_test, ds_info, LR, 
                                #    MOMENTUM, OPT_DECAY, 
                                   BATCH_SIZE, EPOCHS)

history_plain_50 = comile_and_fit(plain_50, ds_train, ds_test, ds_info, LR, 
                                #   MOMENTUM, OPT_DECAY, 
                                  BATCH_SIZE, EPOCHS)

#%% 시각화
plt.subplots(figsize=(15,6))

# ResNet-34, Plain-34 Training Loss 시각화
plt.subplot(121)
plt.plot(history_resnet_34.history['loss'], 'r')
plt.plot(history_plain_34.history['loss'], 'b')
plt.title('Training Loss (ResNet-34 & Plain-34)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['resnet_34', 'plain_34'], loc='upper left')

# ResNet-34, Plain-34 Training Accuracy 시각화
plt.subplot(122)
plt.plot(history_resnet_34.history['accuracy'], 'r')
plt.plot(history_plain_34.history['accuracy'], 'b')
plt.title('Training Accuracy (ResNet-34 & Plain-34)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['resnet_34', 'plain_34'], loc='upper left')

plt.tight_layout()

# 시각화 저장
plt.savefig('./0628_GOINGDEEPER_01/Training_Loss_and_Accuracy_of_34.png')
# plt.show()


plt.subplots(figsize=(15,6))

# ResNet-50, Plain-50 Training Loss 시각화
plt.subplot(121)
plt.plot(history_resnet_50.history['loss'], 'r')
plt.plot(history_plain_50.history['loss'], 'b')
plt.title('Training Loss (ResNet-50 & Plain-50)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['resnet_50', 'plain_50'], loc='upper left')

# ResNet-50, Plain-50 Training Accuracy 시각화
plt.subplot(122)
plt.plot(history_resnet_50.history['accuracy'], 'r')
plt.plot(history_plain_50.history['accuracy'], 'b')
plt.title('Training Accuracy (ResNet-50 & Plain-50)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['resnet_50', 'plain_50'], loc='upper left')

plt.tight_layout()

# 시각화 저장
plt.savefig('./0628_GOINGDEEPER_01/Training_Loss_and_Accuracy_of_50.png')
# plt.show()


plt.subplots(figsize=(15,6))

# ResNet-34, Plain-34, ResNet-50, Plain-50 Training Loss 시각화
plt.subplot(121)
plt.plot(history_resnet_34.history['loss'], 'r')
plt.plot(history_resnet_50.history['loss'], 'b')
plt.plot(history_plain_34.history['loss'], 'y')
plt.plot(history_plain_50.history['loss'], 'g')
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['ResNet_34','ResNet_50','Plain_34','Plain_34'], loc='upper left')

# ResNet-34, Plain-34, ResNet-50, Plain-50 Training Accuracy 시각화
plt.subplot(122)
plt.plot(history_resnet_34.history['accuracy'], 'r')
plt.plot(history_resnet_50.history['accuracy'], 'b')
plt.plot(history_plain_34.history['accuracy'], 'y')
plt.plot(history_plain_50.history['accuracy'], 'g')
plt.title('Training Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['ResNet_34','ResNet_50','Plain_34','Plain_50'], loc='upper left')

plt.tight_layout()

# 시각화 저장
plt.savefig('./0628_GOINGDEEPER_01/Training_Loss_and_Accuracy_of_All.png')
# plt.show()


plt.subplots(figsize=(15,6))

# ResNet-34, Plain-34 Validation Loss 시각화
plt.subplot(121)
plt.plot(history_resnet_34.history['val_loss'], 'r')
plt.plot(history_plain_34.history['val_loss'], 'b')
plt.title('ResNet-34 & Plain-34 Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['resnet_34', 'plain_34'], loc='upper left')

# ResNet-34, Plain-34 Validation Accuracy 시각화
plt.subplot(122)
plt.plot(history_resnet_34.history['val_accuracy'], 'r')
plt.plot(history_plain_34.history['val_accuracy'], 'b')
plt.title('ResNet-34 & Plain-34 Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['resnet_34', 'plain_34'], loc='upper left')

plt.tight_layout()

# 시각화 저장
plt.savefig('./0628_GOINGDEEPER_01/Validation_Loss_and_Accuracy_of_34.png')
# plt.show()


plt.subplots(figsize=(15,6))

# ResNet-34, Plain-34 Validation Loss 시각화
plt.subplot(121)
plt.plot(history_resnet_34.history['val_loss'], 'r')
plt.plot(history_plain_34.history['val_loss'], 'b')
plt.title('ResNet-34 & Plain-34 Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['resnet_34', 'plain_34'], loc='upper left')

# ResNet-34, Plain-34 Validation Accuracy 시각화
plt.subplot(122)
plt.plot(history_resnet_34.history['val_accuracy'], 'r')
plt.plot(history_plain_34.history['val_accuracy'], 'b')
plt.title('ResNet-34 & Plain-34 Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['resnet_34', 'plain_34'], loc='upper left')

plt.tight_layout()

# 시각화 저장
plt.savefig('./0628_GOINGDEEPER_01/Validation_Loss_and_Accuracy_of_34.png')
# plt.show()


plt.subplots(figsize=(15,6))

# ResNet-50, Plain-50 Validation Loss 시각화
plt.subplot(121)
plt.plot(history_resnet_50.history['val_loss'], 'r')
plt.plot(history_plain_50.history['val_loss'], 'b')
plt.title('ResNet-50 & Plain-50 Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['resnet_50', 'plain_50'], loc='upper left')

# ResNet-50, Plain-50 Validation Accuracy 시각화
plt.subplot(122)
plt.plot(history_resnet_50.history['val_accuracy'], 'r')
plt.plot(history_plain_50.history['val_accuracy'], 'b')
plt.title('ResNet-50 & Plain-50 Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['resnet_50', 'plain_50'], loc='upper left')

plt.tight_layout()

# 시각화 저장
plt.savefig('./0628_GOINGDEEPER_01/Validation_Loss_and_Accuracy_of_50.png')
# plt.show()


plt.subplots(figsize=(15,6))

# ResNet-34, Plain-34, ResNet-50, Plain-50 Validation Loss 시각화
plt.subplot(121)
plt.plot(history_resnet_34.history['val_loss'], 'r')
plt.plot(history_resnet_50.history['val_loss'], 'b')
plt.plot(history_plain_34.history['val_loss'], 'y')
plt.plot(history_plain_50.history['val_loss'], 'g')
plt.title('Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['ResNet_34','ResNet_50','Plain_34','Plain_50'], loc='upper left')

# ResNet-34, Plain-34, ResNet-50, Plain-50 Validation Accuracy 시각화
plt.subplot(122)
plt.plot(history_resnet_34.history['val_accuracy'], 'r')
plt.plot(history_resnet_50.history['val_accuracy'], 'b')
plt.plot(history_plain_34.history['val_accuracy'], 'y')
plt.plot(history_plain_50.history['val_accuracy'], 'g')
plt.title('Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['ResNet_34','ResNet_50','Plain_34','Plain_50'], loc='upper left')

plt.tight_layout()

# 시각화 저장
plt.savefig('./0628_GOINGDEEPER_01/Validation_Loss_and_Accuracy_of_All.png')
# plt.show()


#%% 학습 결과 비교
# ResNet, Plain 학습 결과 평균값을 딕셔너리로 저장
model_result = {
    'ResNet-34': [
        round(np.array(history_resnet_34.history['loss']).mean(), 2),
        round(np.array(history_resnet_34.history['accuracy']).mean(), 2),
        round(np.array(history_resnet_34.history['val_loss']).mean(), 2),
        round(np.array(history_resnet_34.history['val_accuracy']).mean(), 2)
    ],
    'Plain-34': [
        round(np.array(history_plain_34.history['loss']).mean(), 2),
        round(np.array(history_plain_34.history['accuracy']).mean(), 2),
        round(np.array(history_plain_34.history['val_loss']).mean(), 2),
        round(np.array(history_plain_34.history['val_accuracy']).mean(), 2)
    ],
    'ResNet-50': [
        round(np.array(history_resnet_50.history['loss']).mean(), 2),
        round(np.array(history_resnet_50.history['accuracy']).mean(), 2),
        round(np.array(history_resnet_50.history['val_loss']).mean(), 2),
        round(np.array(history_resnet_50.history['val_accuracy']).mean(), 2)
    ],
    'Plain-50': [
        round(np.array(history_plain_50.history['loss']).mean(), 2),
        round(np.array(history_plain_50.history['accuracy']).mean(), 2),
        round(np.array(history_plain_50.history['val_loss']).mean(), 2),
        round(np.array(history_plain_50.history['val_accuracy']).mean(), 2)
    ],    
}

# 데이터 프레임 생성
df_model_result = pd.DataFrame(model_result, index=['Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy'])
df_model_result # ResNet-34, Plain-34, ResNet-50, Plain-50 학습 결과

df_val_acc = pd.DataFrame(df_model_result.loc['Validation Accuracy'], columns=['Validation Accuracy'])
df_val_acc # Validation Accuracy 기준으로 결과 확인


