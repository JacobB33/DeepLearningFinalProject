from data.nsd_access import NSDAccess
import scipy.io
import pandas as pd
import numpy as np
def main(subject, processed_data_path='data/processed_data/'):
    nsd = NSDAccess('data/nsd/')
    
    subject_behaviors = pd.DataFrame()
    for i in range(1, 38):
        subject_behaviors = pd.concat([subject_behaviors, nsd.read_behavior(subject, i)])
    
    # It is 1 indexed I think. It has a value of 73000, even though we have the max index of stim_discriptions is 72999
    # These can be used with nsd.read_image_coco_info to get the image information
    annotation_dict = {
        
    }
    stimulus = subject_behaviors['73KID'] -1
    stim_index = 0
    for i in range(1, 38):
        dataset = []
        betas = nsd.read_betas('subj01', session_index=i)
        for index in range(len(betas)):
            beta = betas[index] # numpy array of length 750 which is the voxels of brain activity
            image_id = stimulus[stim_index]
            coco_info = nsd.read_image_coco_info([stimulus[stim_index]]) 
            captions = [info['caption'] for info in coco_info]
            embeddings = get_text_embedding(captions)
            
            
            
            break
        break
        # save the dataaset to a file:
        np.save(processed_data_path + 'subj01_' + str(i), dataset)

        stim_index += len(betas)

    def get_text_embeddings(captions: List[str]):
        pass
    
if __name__ == '__main__':
    main('subj01')