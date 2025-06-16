from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images_from_PIL
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.demo import get_3D_model_from_scene
import PIL.Image
import pandas as pd
import numpy as np

if __name__ == '__main__':
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory

    imageDf = pd.read_csv("/home/soliverosb/dataFor3D/carScene.csv")
    n = 2
    PILImages = [ PIL.Image.open(imageDf['paths'].iloc[i]) for i in range(n) ]
    transformationMatrices = [np.array(imageDf['rotation_matrices'].iloc[i]) for i in range(n)]

    images = load_images_from_PIL(PILImages, size=512)

    pairs = make_pairs(images, scene_graph='complete', prefilter='cyc20', symmetrize=True)
    output = inference(pairs, model, device)


    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()


    for i in range(1, len(poses)):
        homoMat0 = poses[i-1].cpu().detach().numpy()
        homoMat1 = poses[i].cpu().detach().numpy()
        
        
        rotationMat0 = homoMat0[:3, :3]
        transVect0 = homoMat0[:3, 3]

        rotationMat1 = homoMat1[:3, :3]
        transVect1 = homoMat1[:3, 3]

        relativeRotationAccuracy = np.arccos((np.trace(rotationMat0@rotationMat1.T) - 1)/2)

        r = np.arccos((np.trace(transformationMatrices[0]@transformationMatrices[1].T) - 1)/2)

        print(2)

     
        


    # visualize reconstruction
    get_3D_model_from_scene("./", True, scene)