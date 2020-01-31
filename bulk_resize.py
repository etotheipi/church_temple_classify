import cv2
import os
import sys

def bulk_resize(src, dst, targ_dim=512):
    
    if not os.path.exists(dst):
        os.makedirs(dst)
        
    src_trim = src.lstrip('./')
    n_imgs = 0
    for root,subs,files in os.walk(src):
        rel_path = root.lstrip('./')[len(src_trim):].lstrip('/')
        print(rel_path)
        for f in files:
            if not f.lower().split('.')[-1] in ['jpg', 'jpeg', 'bmp', 'gif']:
                continue
                
            src_path = os.path.join(root, f)
            
            try:
                dst_dir = os.path.join(dst, rel_path)
                if not os.path.isdir(dst_dir):
                    os.makedirs(dst_dir)

                dst_path = os.path.join(dst_dir, f)
                orig_img = cv2.imread(src_path)
                ref_size = max(orig_img.shape[:2])
                scale = float(512) / ref_size
                new_sz0 = int(scale * orig_img.shape[0])
                new_sz1 = int(scale * orig_img.shape[1])
                resz_img = cv2.resize(orig_img, (new_sz1, new_sz0))
                print(f'Resizing: {orig_img.shape} -> {resz_img.shape}, to {dst_path}')
                cv2.imwrite(dst_path, resz_img)
                n_imgs += 1
            except Exception as e:
                print(f'Found non image file? {src_path}, error: {str(e)}')
                continue
            
    return n_imgs
            
        
        
if __name__ == '__main__':
    src = sys.argv[1]
    dst = sys.argv[2]
    if not os.path.exists(src):
        raise Exception(f'Invalid source {src}')
    n_imgs = bulk_resize(src, dst)
    print(f'Resized {n_imgs} images')
