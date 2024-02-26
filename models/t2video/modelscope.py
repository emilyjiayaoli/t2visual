import pathlib
from huggingface_hub import snapshot_download
from ..base_model import BaseModel
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
from ..constants import TRANSFORMERS_CACHE
import os
import shutil

class ModelScope(BaseModel):
    """
    ModelScopeDamo
    This class is used to generate videos from descriptions using the ModelScope-DAMO model.
    https://huggingface.co/ali-vilab/modelscope-damo-text-to-video-synthesis
    """
    def __init__(self):
        model_dir = pathlib.Path(os.path.join(TRANSFORMERS_CACHE, 'modelscope_weights'))
        snapshot_download('damo-vilab/modelscope-damo-text-to-video-synthesis', 
                          repo_type='model', local_dir=model_dir)
        self.pipe = pipeline('text-to-video-synthesis', model_dir.as_posix())

    def generate(self, prompt, folder_path="./", filename="modelscope_video.mp4"):
        test_text = {'text': prompt}
        output = self.pipe(test_text)
        output_video_path = output[OutputKeys.OUTPUT_VIDEO]

        # # Assuming output_video_path is a path to the generated video,
        # # move or copy the video to the desired folder_path with the specified filename.
        # final_path = pathlib.Path(folder_path) / filename
        # # If the output video path is not directly saved to the final_path,
        # # you might need to move or copy it from output_video_path to final_path.

        # print(f'Generated video path: {final_path}')
        # return final_path.as_posix()
    
        final_path = pathlib.Path(folder_path) / filename
        
        # Move or copy the video file to the desired location
        if pathlib.Path(output_video_path).exists():
            shutil.move(output_video_path, final_path)
            # Alternatively, to copy the file, you can use:
            # shutil.copy(output_video_path, final_path)
        else:
            print(f'Error: Generated video not found at {output_video_path}')

        print(f'Generated video path: {final_path}')
        return final_path.as_posix()
    
    
