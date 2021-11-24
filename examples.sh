# Reconstruction
python main.py --size=256 --step=5000 --img_path='./samples/true_image/CXR.jpg'
# Image denoising
python main.py --size=256 --step=5000 --img_path='./samples/true_image/CXR.jpg' --ops 'denoising' --ops_fac 0.25
# Attribute direction learning
python main.py --size=256 --step=5000 --img_path='./samples/true_image/CXR.jpg' --attribute 'rotate' --attribute_fac 0 22.5 45