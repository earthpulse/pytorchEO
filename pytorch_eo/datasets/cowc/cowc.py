import os
import math
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
import pytorch_lightning as pl
import xml.etree.ElementTree as ET
from shapely import geometry
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchvision.models.detection import ssd300_vgg16
from torchvision.transforms import ToTensor
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.ssd import SSDHead
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils
from torchvision.models.detection import SSD300_VGG16_Weights
from PIL import Image
from collections import OrderedDict
import dicttoxml
from xml.dom.minidom import parseString
import torchvision.transforms as transforms
 
 ######################################################################
 ## slicing, annotating and pre-processing script for COWC data 

class Slicer:
    def __init__(self, width=544, height=544, overlap=0.1):
        self._overlap = overlap
        self._zero_frac_thresh = 0.2
        self._width = width
        self._height = height

    def process_slice(self, img_pathname, out_path, aoi=None, pad=0):
        image = cv2.imread(img_pathname, 1)
        slices = []
        im_h, im_w = image.shape[:2]
        win_size = self._height * self._width

        if self._height > im_h:
            pad = self._height - im_h
        if self._width > im_w:
            pad = max(pad, self._width - im_w)

        if pad > 0:
            border_color = (0, 0, 0)
            image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=border_color)

        dx = int((1. - self._overlap) * self._width)
        dy = int((1. - self._overlap) * self._height)

        for y in range(0, im_h, dy):
            for x in range(0, im_w, dx):
                y0 = (im_h - self._height) if y + self._height > im_h else y
                x0 = (im_w - self._width) if x + self._width > im_w else x

                if aoi is None or self.isIntersection(aoi, [y0, x0, y0 + self._height, x0 + self._width]):
                    window_c = image[y0:y0 + self._height, x0:x0 + self._width]
                    win_h, win_w = window_c.shape[:2]
                    filename = 'slice_{}_{}_{}_{}_{}_{}'.format(os.path.splitext(os.path.basename(img_pathname))[0], y0, x0, win_h, win_w, pad)
                    out_pathname = os.path.join(out_path, filename + '.jpg')

                    if not os.path.exists(out_pathname):
                        window = cv2.cvtColor(window_c, cv2.COLOR_BGR2GRAY)
                        ret, thresh = cv2.threshold(window, 2, 255, cv2.THRESH_BINARY)
                        non_zero_counts = cv2.countNonZero(thresh)
                        zero_counts = win_size - non_zero_counts
                        zero_frac = float(zero_counts) / win_size

                        if zero_frac >= self._zero_frac_thresh:
                            continue

                        if not os.path.exists(out_path):
                            os.makedirs(out_path)
                        cv2.imwrite(out_pathname, window_c)

                    slices.append({'pathname': out_pathname, 'y0': y0, 'x0': x0, 'win_h': win_h, 'win_w': win_w, 'height': self._height, 'width': self._width, 'pad': pad})

        return slices

    def isIntersection(self, r1, r2):
        p1 = geometry.Polygon([(r1[0],r1[1]), (r1[1],r1[1]),(r1[2],r1[3]),(r1[2],r1[1])])
        p2 = geometry.Polygon([(r2[0],r2[1]), (r2[1],r2[1]),(r2[2],r2[3]),(r2[2],r2[1])])
        return p1.intersects(p2)

class Cowc_annot:

    def __init__( self, size=(3,3) ):

        """
        placeholder
        """

        # list of train and test directories
        self._annotation_suffix = '_Annotated_Cars.png'

        # 15cm resolution
        self._GSD = 0.15
        self._size = ( int( round( ( size[ 0 ] / self._GSD ) / 2 ) ), int ( round ( ( size[ 1 ] / self._GSD ) / 2 ) ) )

        # xml conversion tweak
        self._custom_item_func = lambda x: 'object'

        # create image slicer
        self._slicer = Slicer()

        return


    def process( self, data_path, out_path ):

        """
        create images and annotations for train and validation
        """

        # for each subset
        for subset in [ 'train', 'test' ]:

            # locate all images in data path
            path = os.path.join( data_path, subset ) 
            files = glob.glob( os.path.join( os.path.join( path, '**' ), '*.png' ), recursive=True )    
            files = [ x for x in files if 'Annotated' not in x ]

            # slice up images
            for f in files:

                slices = self._slicer.process_slice ( f, os.path.join( out_path, '{}/images'.format( subset ) ) )
                
                # check annotation image exists
                pathname = os.path.join( f.replace( '.png', self._annotation_suffix ) )
                if os.path.exists( pathname ):

                    # create PASCAL VOC schema for each image slice
                    annotation_image = cv2.imread( pathname )
                    for s in slices:
                        self.getAnnotation( s, annotation_image, os.path.join( out_path, '{}/annotations'.format( subset ) ) )

        return


    def getAnnotation( self, s, annotation_image, out_path, writeback=False, overwrite=True ):

        """
        create annotation xml files encoding bounding box locations
        """

        # create label pathname
        filename = os.path.splitext( os.path.basename( s[ 'pathname' ] ) )[ 0 ] + '.xml' 
        annotation_pathname = os.path.join( out_path, filename )

        if not os.path.exists( annotation_pathname ) or overwrite:

            # get bounding boxes for cars in aoi 
            results, label_locs = self.getBoundingBoxes( s, annotation_image )
            schema = self.getSchema( s, results )

            # create output dir if necessary
            if not os.path.exists( out_path ):
                os.makedirs( out_path )

            # write annotation to xml file
            with open( os.path.join( out_path, filename ), "w+" ) as outfile:

                # parse xml into string
                xml = dicttoxml.dicttoxml( schema, attr_type=False, item_func=self._custom_item_func, custom_root='annotation' ) \
                        .replace(b'<annotation>',b'<annotation verified="yes">') \
                        .replace(b'<items>',b'').replace(b'</items>',b'') \

                dom = parseString( xml )

                # write xml string to file
                outfile.write( dom.toprettyxml() )

            # plot writeback
            if writeback:
                self.drawBoundingBoxes( s[ 'pathname' ], results )

        return


    def getBoundingBoxes( self, s, annotation_image, heading='fixed' ):

        """
        extract bounding boxes around car locations from annotation image
        """

        # process each slice
        records = []

        # extract window from annotation image
        x0 = s[ 'x0' ]; y0 = s[ 'y0' ]
        window = annotation_image[ y0:y0 + s [ 'height' ], x0:x0 + s[ 'width' ] ]

        # find locations of non-zero pixels - add zero rotation column
        label_locs = np.where( window > 0)
        label_locs = np.transpose( np.vstack( [ label_locs[ 0 ], label_locs[ 1 ], np.zeros( len( label_locs[ 0 ] ) ) ]  ) )

        if label_locs.size > 0:

            # create bounding box for annotated car locations
            for loc in label_locs:    
                record = self.getBoundingBox( loc, window.shape )

                # ignore annotated objects close to image edge
                if record:
                    records.append( record )

        return records, label_locs


    def getBoundingBox( self, loc, dims ):

        """
        placeholder
        """
        
        # extrapolate bbox from centroid coords
        record = {}
        yc, xc, angle = loc

        # compute pts along vertical line rotated at mid point
        x0_r, y0_r = self.rotatePoint( xc, yc + self._size[ 1 ], xc, yc, math.radians( angle ) ) 
        x1_r, y1_r = self.rotatePoint( xc, yc - self._size[ 1 ], xc, yc, math.radians( angle ) ) 

        # compute corner pts orthogonal to rotated line end points
        corner = np.empty( (4, 2), float )

        corner[ 0 ] = self.rotatePoint( x0_r, y0_r + self._size[ 0 ], x0_r, y0_r, math.radians( angle + 90.0 ) )
        corner[ 1 ] = self.rotatePoint( x0_r, y0_r - self._size[ 0 ], x0_r, y0_r, math.radians( angle + 90.0 ) )

        corner[ 2 ] = self.rotatePoint( x1_r, y1_r + self._size[ 0 ], x1_r, y1_r, math.radians( angle + 90.0 ) )
        corner[ 3 ] = self.rotatePoint( x1_r, y1_r - self._size[ 0 ], x1_r, y1_r, math.radians( angle + 90.0 ) )

        # get min and max coordinates for bbox
        x_min = np.amin( corner[ :, 0 ] ); x_max = np.amax( corner[ :, 0 ] )
        y_min = np.amin( corner[ :, 1 ] ); y_max = np.amax( corner[ :, 1 ] )

        # check limits
        x_min_c = max( 0, x_min ); y_min_c = max( 0, y_min )
        x_max_c = min( x_max, dims[1] - 1 ); y_max_c = min( y_max, dims[0] - 1 )        

        area = ( x_max - x_min ) * ( y_max - y_min )
        area_c = ( x_max_c - x_min_c ) * ( y_max_c - y_min_c )

        # only retain bboxes not constrained by image edges
        if area_c / area > 0.95:

            record[ 'bbox' ] = [ x_min_c, y_min_c, x_max_c, y_max_c ]

            # readjust perimeter points
            corner[ :, 0 ] = np.where( corner[ :, 0 ] < 0.0, 0.0, corner[ :, 0 ] )
            corner[ :, 0 ] = np.where( corner[ :, 0 ] > dims[1] - 1, dims[1] - 1, corner[ :, 0 ] )

            corner[ :, 1 ] = np.where( corner[ :, 1 ] < 0.0, 0.0, corner[ :, 1 ] )
            corner[ :, 1 ] = np.where( corner[ :, 1 ] > dims[0] - 1, dims[0] - 1, corner[ :, 1 ] )

            # minimise distance between points
            d1 = np.linalg.norm( corner[ 1 ] - corner[ 2 ] ); d2 = np.linalg.norm( corner[ 1 ] - corner[ 3 ] )
            if d1 > d2:
                corner[ [ 2, 3 ] ] = corner[ [ 3, 2 ] ]

            record[ 'corner' ] = list( corner.flatten() )

        return record


    def rotatePoint( self, x, y, xc, yc, angle ):

        """
        compute rotation of point around origin
        """

        # Rotate point counterclockwise by a given angle around a given origin.
        qx = xc + math.cos(angle) * (x - xc) - math.sin(angle) * (y - yc)
        qy = yc + math.sin(angle) * (x - xc) + math.cos(angle) * (y - yc)

        return qx, qy


    def getSchema( self, s, records ):

        """
        convert annotation into ordered list for conversion into PASCAL VOC schema
        """

        # convert to PASCAL VOC annotation schema
        object_list = []
        for record in records:

            bbox = record[ 'bbox' ]; #corner = record[ 'corner' ]
            object_list.append( OrderedDict ( {     'name' : 'car',
                                                    'pose': 'Topdown',
                                                    'truncated' : 0,
                                                    'difficult': 0,
                                                    'bndbox': {'xmin': bbox[ 0 ], 'ymin': bbox[ 1 ], 'xmax': bbox[ 2 ], 'ymax': bbox[ 3 ] }
                                                    #'segmentation' : ','.join( (str(pt) for pt in corner ) ) 
                                            } ) )

        # return full schema as dictionary
        return OrderedDict ( {  'folder' : 'images',
                                'filename' : os.path.basename( s[ 'pathname' ] ),
                                'path' : os.path.dirname( s[ 'pathname' ] ),
                                'source' : { 'database': 'cowc' },
                                'size' : { 'width' : s[ 'width' ], 'height' : s[ 'height' ], 'depth' : 3 },
                                'segmented' : 0,
                                'items' : object_list } )


    def drawBoundingBoxes( self, pathname, records ):

        """
        placeholder
        """

        # no action if no bboxes
        if len ( records ) > 0:

            # load image
            img = cv2.imread( pathname )                                  
            height = img.shape[0]; width = img.shape[ 1 ]
                    
            # show image
            plt.imshow( cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ); ax = plt.gca()
            fig = plt.gcf(); fig.canvas.set_window_title( os.path.basename( pathname ) )
            print ( pathname )

            # draw bbox lines
            colors = [ 'r', 'g', 'y', 'b', 'm', 'c' ]; idx = 0
            for record in records:    

                x0, y0, x1, y1 = record[ 'bbox' ]

                color = colors[ idx ] + '-'
                idx = idx + 1 if idx + 1 < len ( colors ) else 0

                ax.plot( [ x0, x1 ], [ y0, y0 ], color )
                ax.plot( [ x0, x1 ], [ y1, y1 ], color )
                ax.plot( [ x0, x0 ], [ y0, y1 ], color )
                ax.plot( [ x1, x1 ], [ y0, y1 ], color )

                """
                # get run length encoding from perimeter points string
                rl_encoding = mask.frPyObjects( [ record[ 'corner' ] ] , height, width )

                binary_mask = mask.decode( rl_encoding )
                binary_mask = np.amax(binary_mask, axis=2)

                masked = np.ma.masked_where(binary_mask == 0, binary_mask )
                ax.imshow( masked, 'jet', interpolation='None', alpha=0.5 )
                """

            plt.show()

        return
    

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, images, annotations, transform=None):
        self.images = images
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]['file_name']
        image = Image.open(img_path).convert("RGB")
        boxes = self.images[idx]['boxes']

        if self.transform:
            image = self.transform(image)

        # # Provide a dummy box with positive width and height, outside image bounds
        if len(boxes) == 0: 
            boxes = [[-1, -1, -0.5, -0.5]]  
            labels = torch.tensor([0], dtype=torch.int64)  # Label is 0 if there are no bounding boxes
        else:
            labels = torch.tensor([1] * len(boxes), dtype=torch.int64)  # Label is 1 if there are bounding boxes

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # if len(boxes) > 0:  # Only return images with bounding boxes
        #     if self.transform:
        #         image = self.transform(image)

        #     boxes = torch.as_tensor(boxes, dtype=torch.float32)
        #     labels = torch.tensor([1] * len(boxes), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        return image, target
        # Ensure idx stays within bounds
        # while idx < len(self.images):
        #     img_path = self.images[idx]['file_name']
        #     image = Image.open(img_path).convert("RGB")
        #     boxes = self.images[idx]['boxes']

        #     if len(boxes) > 0:  # Only return images with bounding boxes
        #         if self.transform:
        #             image = self.transform(image)

        #         boxes = torch.as_tensor(boxes, dtype=torch.float32)
        #         labels = torch.ones((len(boxes),), dtype=torch.int64)  # All labels are 1 (e.g., 'car')

        #         target = {}
        #         target["boxes"] = boxes
        #         target["labels"] = labels

        #         return image, target
            
        #     # If no bounding boxes, move to the next index
        #     idx += 1

    

'''
class ObjectDetectionModel(pl.LightningModule):
    def __init__(self):
        super(ObjectDetectionModel, self).__init__()
        # Load the Torchvision pretrained model.
    
        self.model = ssd300_vgg16(pretrained=True)
        num_classes = 2  # 1 class (object) + background

        # Update the classification head
        self.model.head.classification_head.num_classes = num_classes

    def forward(self, images, targets=None):
        if self.training:
            return self.model(images, targets)
        else:
            return self.model(images)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        return optimizer
'''
def parse_voc_annotation(annot_folder, image_folder):
    all_images = []
    all_annotations = []

    for annot_file in os.listdir(annot_folder):
        if annot_file.endswith('.xml'):
            annot_path = os.path.join(annot_folder, annot_file)

            tree = ET.parse(annot_path)
            root = tree.getroot()

            image_info = {}
            image_info['file_name'] = os.path.join(image_folder, root.find('filename').text)
            boxes = []

            for obj in root.findall('object'):
                bbox = obj.find('bndbox')
                box = [int(float(bbox.find('xmin').text)), int(float(bbox.find('ymin').text)),
                    int(float(bbox.find('xmax').text)), int(float(bbox.find('ymax').text))]
                boxes.append(box)

            image_info['boxes'] = boxes
            all_images.append(image_info)
            all_annotations.append(boxes)

    return all_images, all_annotations

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return tuple(zip(*batch))

class ObjectDetectionDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=4, transform=None):
        super(ObjectDetectionDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
        ])
        self.val_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.train_dataset = None  # Initialize these later in setup()
        self.val_dataset = None

    def setup(self, stage=None):
        train_images, train_annotations = parse_voc_annotation(
            os.path.join(self.data_dir, 'train/annotations'), 
            os.path.join(self.data_dir, 'train/images')
        )
        val_images, val_annotations = parse_voc_annotation(
            os.path.join(self.data_dir, 'test/annotations'), 
            os.path.join(self.data_dir, 'test/images')
        )

        self.train_dataset = VOCDataset(train_images, train_annotations, transform=self.train_transforms)
        self.val_dataset = VOCDataset(val_images, val_annotations, transform=self.val_transforms)

    def get_dataloader(self, ds, batch_size=None, shuffle=False, collate_fn=None):
            return torch.utils.data.DataLoader(
                ds,
                batch_size=self.batch_size if batch_size is None else batch_size,
                shuffle=shuffle,
                collate_fn=collate_fn,
            )

    def train_dataloader(self):
        return self.get_dataloader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)

    def val_dataloader(self):
        if self.val_dataset is not None:
            return self.get_dataloader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        else:
            return None

# if __name__ == '__main__':
#     data_module = ObjectDetectionDataModule(data_dir='/fastdata/COWC/v1/datasets/ground_truth_sets/processed')
#     data_module.setup()



# data_module = ObjectDetectionDataModule(data_dir='/fastdata/COWC/v1/datasets/ground_truth_sets/processed', batch_size=4)

# data_module.setup()