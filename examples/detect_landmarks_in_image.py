import face_alignment
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import io

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Run the 3D face alignment on a test image.
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
    enable_cuda=True, flip_input=False)

# fa.make_rct_files("Databases/FEI")
# result_list = []

# fa.use_STN("Models/12-06_Theta.pth")
# fa.train_STN("Databases/lfpw/trainset_normals_only", 1, "Models/12-06_Theta2.pth")
fa.use_STN("Models/12-06_Theta3.pth")

fa.train_STN("Databases/lfpw/trainset", 1, "Models/12-06_ThetaThenFan.pth")
# fa.train_STN("Databases/10W", 4, "Models/test.pth")
fa.use_STN("Models/12-06_ThetaThenFan.pth")
# fa.use_STN_from_caffe()

result_list = fa.process_folder("Databases/lfpw/testset", 1)
# result_list = fa.process_folder("Databases/10W", 4)

for [image_name, preds_all] in result_list:
    landmarks, gt_landmarks, proposal_img, frontal_img, _, _ = preds_all
    preds = landmarks[-1]          # [-1]: Use only the last face detected when there are multiple faces in one picture (all_faces=True)
    gts = gt_landmarks[-1]
    input = io.imread(image_name)
    fig = plt.figure(figsize=(10,10), tight_layout=True)

    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(input)
    ax.plot(gts[0:17,0] ,gts[0:17,1], marker='o',markersize=4,linestyle='-',color='b',lw=1)
    ax.plot(gts[17:22,0],gts[17:22,1],marker='o',markersize=4,linestyle='-',color='b',lw=1)
    ax.plot(gts[22:27,0],gts[22:27,1],marker='o',markersize=4,linestyle='-',color='b',lw=1)
    ax.plot(gts[27:31,0],gts[27:31,1],marker='o',markersize=4,linestyle='-',color='b',lw=1)
    ax.plot(gts[31:36,0],gts[31:36,1],marker='o',markersize=4,linestyle='-',color='b',lw=1)
    ax.plot(gts[36:42,0],gts[36:42,1],marker='o',markersize=4,linestyle='-',color='b',lw=1)
    ax.plot(gts[42:48,0],gts[42:48,1],marker='o',markersize=4,linestyle='-',color='b',lw=1)
    ax.plot(gts[48:60,0],gts[48:60,1],marker='o',markersize=4,linestyle='-',color='b',lw=1)
    ax.plot(gts[60:68,0],gts[60:68,1],marker='o',markersize=4,linestyle='-',color='b',lw=1)
    ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.axis('off')

    if fa.landmarks_type == face_alignment.LandmarksType._3D:
        ax = fig.add_subplot(2, 2, 2, projection='3d')
        surf = ax.scatter(preds[:,0]*1.2,preds[:,1],preds[:,2],c="cyan", alpha=1.0, edgecolor='b')
        ax.plot3D(preds[:17,0]*1.2,preds[:17,1], preds[:17,2], color='blue' )
        ax.plot3D(preds[17:22,0]*1.2,preds[17:22,1],preds[17:22,2], color='blue')
        ax.plot3D(preds[22:27,0]*1.2,preds[22:27,1],preds[22:27,2], color='blue')
        ax.plot3D(preds[27:31,0]*1.2,preds[27:31,1],preds[27:31,2], color='blue')
        ax.plot3D(preds[31:36,0]*1.2,preds[31:36,1],preds[31:36,2], color='blue')
        ax.plot3D(preds[36:42,0]*1.2,preds[36:42,1],preds[36:42,2], color='blue')
        ax.plot3D(preds[42:48,0]*1.2,preds[42:48,1],preds[42:48,2], color='blue')
        ax.plot3D(preds[48:,0]*1.2,preds[48:,1],preds[48:,2], color='blue' )
        ax.view_init(elev=90., azim=90.)
        ax.set_xlim(ax.get_xlim()[::-1])

    if proposal_img:
        ax = fig.add_subplot(2, 2, 3)
        ax.imshow(proposal_img[-1])
        ax.axis('off')
    if frontal_img:
        ax = fig.add_subplot(2, 2, 4)
        ax.imshow(frontal_img[-1])
        ax.axis('off')

    plt.show()