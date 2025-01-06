# AIhw
# COVID-19：使用深度学习的医学诊断
南華大學_跨領域-人工智慧_Exam

11024103林仁楚 11024227邱胤睿
## 介绍
正在進行的名為COVID-19的全球大流行是由SARS-COV-2引起的，
該病毒傳播迅速並發生變異，引發了幾波疫情，主要影響第三世界和發展中國家。隨著世界各國政府試圖控制傳播，受影響的人數正穩定上升。

![ai1](https://github.com/zipo0505/AIhw/blob/main/ai1.png)

本文將使用CoronaHack-Chest X 光資料集。
它包含胸部X 光影像，我們必須找到受冠狀病毒影響的影像。

我們之前談到的SARS-COV-2 是主要影響呼吸系統的病毒類型，因此胸部X 光是我們可以用來識別受影響肺部的重要影像方法之一。這是一個並排比較：

![ai2](https://github.com/zipo0505/AIhw/blob/main/ai2.jpeg)

如你所見，COVID-19 肺炎如何吞噬整個肺部，並且比細菌和病毒類型的肺炎更危險。

本文，將使用深度學習和遷移學習對受Covid-19影響的肺部的X 光影像進行分類和識別。
## 導入庫和載入數據
    import matplotlib.pyplot as plt
    import seaborn as sns
    %matplotlib inline
    import numpy as np
    import pandas as pd
    sns.set()
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import  *
    from tensorflow.keras.optimizers import Adam, SGD, RMSprop
    from tensorflow.keras.applications import DenseNet121, VGG19, ResNet50

    import PIL.Image
    import matplotlib.pyplot as mpimg
    import os
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
    from tensorflow.keras.preprocessing import image

    from tqdm import tqdm
    import warnings
    warnings.filterwarnings("ignore")

    from sklearn.utils import shuffle

    train_df = pd.read_csv('../input/corona hack-chest-xray dataset/Chest_xray_Corona_Metadata.csv')
    train_df.shape
    > (5910, 6)

    train_df.head(5)
    
![ai03](https://github.com/zipo0505/AIhw/blob/main/ai3.jepg)

    train_df.info()
    
![ai4](https://github.com/zipo0505/AIhw/blob/main/ai4.jepg)

## 處理缺失值
    missing_vals = train_df.isnull().sum()
    missing_vals.plot(kind = 'bar')
    
![ai5](https://github.com/zipo0505/AIhw/blob/main/ai5.jepg)

    train_df.dropna(how = 'all')
    train_df.isnull().sum()
    
![ai6](https://github.com/zipo0505/AIhw/blob/main/ai6.jepg)

    train_df.fillna('unknown', inplace=True)
    train_df.isnull().sum()
    
![ai7](https://github.com/zipo0505/AIhw/blob/main/ai7.jepg)

    train_data = train_df[train_df['Dataset_type'] == 'TRAIN']
    test_data = train_df[train_df['Dataset_type'] == 'TEST']
    assert train_data.shape[0] + test_data.shape[0] == train_df.shape[0]
    print(f"Shape of train data : {train_data.shape}")
    print(f"Shape of test data : {test_data.shape}")
    test_data.sample(10)
    
![ai8](https://github.com/zipo0505/AIhw/blob/main/ai8.jepg)

我們將用“unknown”填充缺失值。

    print((train_df['Label_1_Virus_category']).value_counts())
    print('--------------------------')
    print((train_df['Label_2_Virus_category']).value_counts())
    
![ai9](https://github.com/zipo0505/AIhw/blob/main/ai9.jepg)

因此標籤2 類別包含COVID-19案例
## 顯示影像
    test_img_dir = '/kaggle/input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test'
    train_img_dir = '/kaggle/input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train'

    sample_train_images = list(os.walk(train_img_dir))[0][2][:8]
    sample_train_images = list(map(lambda x: os.path.join(train_img_dir, x), sample_train_images))

    sample_test_images = list(os.walk(test_img_dir))[0][2][:8]
    sample_test_images = list(map(lambda x: os.path.join(test_img_dir, x), sample_test_images))

    plt.figure(figsize = (10,10))
    for iterator, filename in enumerate(sample_train_images):
        image = PIL.Image.open(filename)
        plt.subplot(4,2,iterator+1)
        plt.imshow(image, cmap=plt.cm.bone)

    plt.tight_layout()
    
![ai10](https://github.com/zipo0505/AIhw/blob/main/ai10.jepg)
## 視覺化
    plt.figure(figsize=(15,10))
    sns.countplot(train_data['Label_2_Virus_category']);
![ai11](https://github.com/zipo0505/AIhw/blob/main/ai11.jepg)
## 對於COVID-19 病例
    fig, ax = plt.subplots(4, 2, figsize=(15, 10))


    covid_path = train_data[train_data['Label_2_Virus_category']=='COVID-19']['X_ray_image_name'].values

    sample_covid_path = covid_path[:4]
    sample_covid_path = list(map(lambda x: os.path.join(train_img_dir, x), sample_covid_path))

    for row, file in enumerate(sample_covid_path):
        image = plt.imread(file)
        ax[row, 0].imshow(image, cmap=plt.cm.bone)
        ax[row, 1].hist(image.ravel(), 256, [0,256])
        ax[row, 0].axis('off')
        if row == 0:
            ax[row, 0].set_title('Images')
            ax[row, 1].set_title('Histograms')
    fig.suptitle('Label 2 Virus Category = COVID-19', size=16)
    plt.show()
![ai12](https://github.com/zipo0505/AIhw/blob/main/ai12.jepg)
## 對於正常情況
    fig, ax = plt.subplots(4, 2, figsize=(15, 10))


    normal_path = train_data[train_data['Label']=='Normal']['X_ray_image_name'].values

    sample_normal_path = normal_path[:4]
    sample_normal_path = list(map(lambda x: os.path.join(train_img_dir, x), sample_normal_path))

    for row, file in enumerate(sample_normal_path):
        image = plt.imread(file)
        ax[row, 0].imshow(image, cmap=plt.cm.bone)
        ax[row, 1].hist(image.ravel(), 256, [0,256])
        ax[row, 0].axis('off')
        if row == 0:
            ax[row, 0].set_title('Images')
            ax[row, 1].set_title('Histograms')
    fig.suptitle('Label = NORMAL', size=16)
    plt.show()
![ai13](https://github.com/zipo0505/AIhw/blob/main/ai13.jepg)

    final_train_data = train_data[(train_data['Label'] == 'Normal') | 
                                  ((train_data['Label'] == 'Pnemonia') &
                                   (train_data['Label_2_Virus_category'] == 'COVID-19'))]

    final_train_data['class'] = final_train_data.Label.apply(lambda x: 'negative' if x=='Normal' else 'positive')
    test_data['class'] = test_data.Label.apply(lambda x: 'negative' if x=='Normal' else 'positive')

    final_train_data['target'] = final_train_data.Label.apply(lambda x: 0 if x=='Normal' else 1)
    test_data['target'] = test_data.Label.apply(lambda x: 0 if x=='Normal' else 1)

    final_train_data = final_train_data[['X_ray_image_name', 'class', 'target', 'Label_2_Virus_category']]
    final_test_data = test_data[['X_ray_image_name', 'class', 'target']]

    test_data['Label'].value_counts()

## 數據增強
    datagen =  ImageDataGenerator(
      shear_range=0.2,
      zoom_range=0.2,
    )

    def read_img(filename, size, path):
        img = image.load_img(os.path.join(path, filename), target_size=size)
        #convert image to array
        img = image.img_to_array(img) / 255
        return img

    samp_img = read_img(final_train_data['X_ray_image_name'][0],
                                     (255,255),
                                     train_img_path)

    plt.figure(figsize=(10,10))
    plt.suptitle('Data Augmentation', fontsize=28)

    i = 0


    for batch in datagen.flow(tf.expand_dims(samp_img,0), batch_size=6):
        plt.subplot(3, 3, i+1)
        plt.grid(False)
        plt.imshow(batch.reshape(255, 255, 3));
    
        if i == 8:
            break
        i += 1
    
    plt.show();
    
![ai14](https://github.com/zipo0505/AIhw/blob/main/ai14.jepg)

    corona_df = final_train_data[final_train_data['Label_2_Virus_category'] == 'COVID-19']
    with_corona_augmented = []


    def augment(name):
        img = read_img(name, (255,255), train_img_path)
        i = 0
        for batch in tqdm(datagen.flow(tf.expand_dims(img, 0), batch_size=32)):
            with_corona_augmented.append(tf.squeeze(batch).numpy())
            if i == 20:
                break
            i =i+1


    corona_df['X_ray_image_name'].apply(augment)
### 注意：
輸出太長，無法包含在文章中。這是其中的一小部分。

![ai15](https://github.com/zipo0505/AIhw/blob/main/ai15.jepg)

    train_arrays = [] 
    final_train_data['X_ray_image_name'].apply(lambda x: train_arrays.append(read_img(x, (255,255), train_img_dir)))
    test_arrays = []
    final_test_data['X_ray_image_name'].apply(lambda x: test_arrays.append(read_img(x, (255,255), test_img_dir)))

    print(len(train_arrays))
    print(len(test_arrays))

    y_train = np.concatenate((np.int64(final_train_data['target'].values), np.ones(len(with_corona_augmented), dtype=np.int64)))
## 將所有資料轉換為張量
    train_tensors = tf.convert_to_tensor(np.concatenate((np.array(train_arrays), np.array(with_corona_augmented))))
    test_tensors  = tf.convert_to_tensor(np.array(test_arrays))
    y_train_tensor = tf.convert_to_tensor(y_train)
    y_test_tensor = tf.convert_to_tensor(final_test_data['target'].values)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_tensors, y_train_tensor))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_tensors, y_test_tensor))

    for i,l in train_dataset.take(1):
        plt.imshow(i);
![ai16](https://github.com/zipo0505/AIhw/blob/main/ai16.jepg)
## 產生批次
    BATCH_SIZE = 16
    BUFFER = 1000

    train_batches = train_dataset.shuffle(BUFFER).batch(BATCH_SIZE)
    test_batches = test_dataset.batch(BATCH_SIZE)

    for i,l in train_batches.take(1):
        print('Train Shape per Batch: ',i.shape);
    for i,l in test_batches.take(1):
        print('Test Shape per Batch: ',i.shape);

## 使用ResNet50 進行遷移學習
    INPUT_SHAPE = (255,255,3) 

    base_model = tf.keras.applications.ResNet50(input_shape= INPUT_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    # We set it to False because we don't want to mess with the pretrained weights of the model.
    base_model.trainable = False
    
現在我們的遷移學習成功了！ ！

    for i,l in train_batches.take(1):
        pass
    base_model(i).shape
    > TensorShape([16, 8, 8, 2048])
## 為影像分類添加密集層
    model = Sequential()
    model.add(base_model)
    model.add(Layers.GlobalAveragePooling2D())
    model.add(Layers.Dense(128))
    model.add(Layers.Dropout(0.2))
    model.add(Layers.Dense(1, activation = 'sigmoid'))
    model.summary()
    
![ai17](https://github.com/zipo0505/AIhw/blob/main/ai17.jepg)

    callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)


    model.compile(optimizer='adam',
                  loss = 'binary_crossentropy',
                  metrics=['accuracy'])
## 預測
    model.fit(train_batches, epochs=10, validation_data=test_batches, callbacks=[callbacks])
    
![ai18](https://github.com/zipo0505/AIhw/blob/main/ai18.jepg)

    pred = model.predict_classes(np.array(test_arrays))

    # classification report
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(test_data['target'], pred.flatten()))
    
![ai19](https://github.com/zipo0505/AIhw/blob/main/ai19.jepg)

所以正如你所看到的，預測還不錯。我們將繪製一個混淆矩陣來視覺化我們模型的表現：

    con_mat = confusion_matrix(test_data['target'], pred.flatten())
    plt.figure(figsize = (10,10))
    plt.title('CONFUSION MATRIX')
    sns.heatmap(con_mat, cmap='cividis',
                yticklabels=['Negative', 'Positive'],
                xticklabels=['Negative', 'Positive'],
                annot=True);
                
![ai20](https://github.com/zipo0505/AIhw/blob/main/ai20.jepg)
