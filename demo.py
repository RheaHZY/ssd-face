import numpy as np
import os,cv2,platform,sys,argparse,time

if platform.system() == "Windows":
    caffe_root = "D:/CNN/ssd"
else:
    caffe_root = os.path.expanduser('~') + "/CNN/ssd"
sys.path.insert(0, caffe_root+'/python')
import caffe

WINDOWSNAME="ssd-face"
LABEL_NAME="Face"

def drawDetectionResult(image,result,colors=None,cost=None):
    if image is None:
        return image
    height,width,c=image.shape
    show=image.copy()
    for item in result:
        xmin = int(round(item[0] * width))
        ymin = int(round(item[1] * height))
        xmax = int(round(item[2] * width))
        ymax = int(round(item[3] * height))
        if colors is None:
            cv2.putText(show,LABEL_NAME,(xmin,ymin), cv2.FONT_ITALIC,1,(0,0,255))
            cv2.rectangle(show,(xmin,ymin),(xmax,ymax),(255,0,0))
        else:
            color=colors[int(round(item[4]))]
            color=[c *256 for c in color]
            cv2.putText(show,LABEL_NAME,(xmin,ymin), cv2.FONT_ITALIC,1,color)
            cv2.rectangle(show,(xmin,ymin),(xmax,ymax),color)

    if not cost is None:
        cv2.putText(show,cost,(0,40),3,1,(0,0,255))

    return show

class SSDDetection:
    def __init__(self,model_def, model_weights):
        if not os.path.exists(model_weights):
            print(model_weights + " does not exist,")
            exit()
        self.net = caffe.Net(model_def,model_weights,caffe.TEST)
        self.height = self.net.blobs['data'].shape[2]
        self.width = self.net.blobs['data'].shape[3]

    def preprocess(self,src):
        img = cv2.resize(src, (self.height,self.width))
        img = np.array(img, dtype=np.float32)
        img -= np.array((104, 117, 123)) 
        return img

    def detectAndDraw(self,img):
        start=time.time()
        result=self.detect(img)
        end=time.time()
        cost="%0.2fms" %((end-start)*1000)
        show=drawDetectionResult(img,result)
        return show
    def detect(self,img, conf_thresh=0.5, topn=10):
        self.net.blobs['data'].data[...] = self.preprocess(img).transpose((2, 0, 1)) 
        detections = self.net.forward()['detection_out']
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        result = []
        for i in range(min(topn, top_conf.shape[0])):
            xmin = top_xmin[i] # xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = top_ymin[i] # ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = top_xmax[i] # xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = top_ymax[i] # ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            result.append([xmin, ymin, xmax, ymax, label, score])
        return result

def testdir(detection,dir):
    files=os.listdir(dir)
    for file in files:
        imgfile=dir+"/"+file
        img = cv2.imread(imgfile)
        show=detection.detectAndDraw(img)
        cv2.imshow(WINDOWSNAME,show)
        cv2.waitKey()

def testcamera(detection,index=0):
    cap=cv2.VideoCapture(index)
    while True:
        ret,img=cap.read()
        if not ret:
            break
        show=detection.detectAndDraw(img)
        cv2.imshow(WINDOWSNAME,show)
        cv2.waitKey(1)

def main(args):
    '''main '''
    caffe.set_device(args.gpu_id)
    caffe.set_mode_gpu()
    model_def = args.modeldir + '/face_deploy.prototxt'
    model_weights = args.modeldir + '/VGG_Face2017_SSD_300x300_iter_120000.caffemodel'
    detection = SSDDetection(model_def, model_weights)
    #testcamera(detection)
    testdir(detection,"images")

def get_args():
    '''parse args'''    
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', default="models", help='dataset')
    return parser.parse_args()

if __name__ == '__main__':
    main(get_args())