# /home/liuxz/work/dlws/models/vgg16/VGG_2014_16.prototxt
#  /home/liuxz/work/dlws/models/vgg16/VGG16_Coffe/VGG_ILSVRC_16_layers_deploy.prototxt
python convert.py  /home/liuxz/work/dlws/models/vgg16/VGG_2014_16.prototxt --caffemodel /home/liuxz/work/dlws/models/vgg16/VGG16_Coffe/VGG_ILSVRC_16_layers.caffemodel --code-output-path=Vgg16Tensorflow.py --data-output-path=Vgg16Tensorflow_data.npy 
