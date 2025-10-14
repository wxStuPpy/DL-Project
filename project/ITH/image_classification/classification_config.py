#数据预处理配置
IMG_PATH='../common/dataset/'
LABELS_PATH='../common/fashion-labels.csv'
IMG_HEIGHT=64
IMG_WIDTH=64

#随机性和数据集划分
SEED=42
TRAIN_SIZE=0.75
TEST_SIZE=0.25


#相关超参数
LEARNING_RATE=0.001
EPOCHS=20
TRAIN_BATCH_SIZE=128
TEST_BATCH_SIZE=128

#模块名称和保存模型参数的文件
PROJECT_PACKAGE_NAME='image_classification'
CLASSIFIER_MODEL_NAME='classifier.pt'

#数字标签对应分类名称的字典
classification_names = {
    0: 'clothes',
    1: 'shoe',
    2: 'bag',
    3: 'pants',
    4: 'watch'
}