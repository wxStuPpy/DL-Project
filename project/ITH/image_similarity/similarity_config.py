#数据预处理配置
IMG_PATH='../common/dataset/'
IMG_HEIGHT=64
IMG_WIDTH=64

#随机性和数据集划分
SEED=42
TRAIN_SIZE=0.75
TEST_SIZE=0.25

#相关超参数
LEARNING_RATE=0.001
EPOCHS=20
TRAIN_BATCH_SIZE=64
TEST_BATCH_SIZE=64
FULL_BATCH_SIZE=64

#模块名称和保存模型参数的文件
PROJECT_PACKAGE_NAME='image_similarity'
ENCODE_MODEL_NAME='deep_encoder.pt'
DECODE_MODEL_NAME='deep_decoder.pt'
EMBEDDING_NAME='data_embedding.npy'