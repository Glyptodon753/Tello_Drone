import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import cv2


def cam(model, x, threshold=0.3, classes=('Fire', 'Neutral', 'Smoke')):
    x = np.expand_dims(x, axis=0)
    print('aaa')
    last_conv_layer = model.get_layer('conv2d_3')
    predict = model.predict(x)
    class_idx = np.argmax(predict[0])
    # print(predict)
    # print(classes[int(class_idx)])

    iterate = K.function([model.input], [last_conv_layer.output[0]])
    conv_layer_output_value = np.squeeze(iterate([x]), axis=0)

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    # plt.matshow(heatmap)
    # plt.show()

    heatmap = cv2.resize(heatmap, (x.shape[2], x.shape[1]))
    class_area = np.sum(heatmap > threshold)

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    x = np.squeeze(x, axis=0) * 255.0
    x = x.astype('uint8')[:, :, ::-1]

    superimposed_img = heatmap * 0.4 + x
    # cv2.imwrite('cam.jpg', superimposed_img)
    superimposed_img = np.minimum(superimposed_img[:, :, ::-1] / 255.0, 1)

    return classes[int(class_idx)], class_area, heatmap, superimposed_img


def cam2(model, x, threshold=0.3, classes=('Fire', 'Neutral', 'Smoke')):
    x = np.expand_dims(x, axis=0)

    last_conv_layer = model.get_layer('conv2d_3')
    predict = model.predict(x)
    class_idx = np.argmax(predict[0])
    # print(predict)
    # print(classes[int(class_idx)])

    class_output = model.output[:, class_idx]
    gap_weights = model.get_layer("global_average_pooling2d_1")

    grads = K.gradients(class_output, gap_weights.output)[0]
    iterate = K.function([model.input], [grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    pooled_grads_value = np.squeeze(pooled_grads_value, axis=0)
    for i in range(len(pooled_grads_value)):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    # plt.matshow(heatmap)
    # plt.show()

    heatmap = cv2.resize(heatmap, (x.shape[2], x.shape[1]))
    class_area = np.sum(heatmap > threshold)

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    x = np.squeeze(x, axis=0) * 255.0
    x = x.astype('uint8')[:, :, ::-1]

    superimposed_img = heatmap * 0.4 + x
    cv2.imwrite('cam.jpg', superimposed_img)
    superimposed_img = np.minimum(superimposed_img[:, :, ::-1] / 255.0, 1)

    return classes[int(class_idx)], class_area, heatmap, superimposed_img
