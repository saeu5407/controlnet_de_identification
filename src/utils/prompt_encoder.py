from typing import Optional, Union, List, Any, Callable, Dict
import torch
from transformers import AutoTokenizer, CLIPTextModel

class Clip():

    def __init__(self,
                 model_name_or_path: Optional[str],
                 device: Union[torch.device, None] = None,
                 requires_grad_: Optional[bool] = False
                 ):
        super(Clip, self).__init__()
        self.model_name = model_name_or_path
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, subfolder="tokenizer",)
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path=self.model_name,
                                                          subfolder="text_encoder",).to(self.device)
        self.text_encoder.requires_grad_(requires_grad_)

    def encode_propmt(self,
                      prompt: Optional[str],
                      negative_prompt: Optional[str] = "",
                      do_classifier_free_guidance: Optional[bool] = False
                      ):
        text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors="pt")
        negative_text_input = self.tokenizer(negative_prompt, padding="max_length",
                                             max_length=self.tokenizer.model_max_length,
                                             truncation=True, return_tensors="pt")
        with torch.no_grad():
            prompt_embeds = self.text_encoder(text_input.input_ids.to(self.device))[0]
            negative_text_embeds = self.text_encoder(negative_text_input.input_ids.to(self.device))[0]
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_text_embeds, prompt_embeds])

        return prompt_embeds

if __name__ == '__main__':

    clip = Clip(model_name_or_path="Linaqruf/anything-v3.0")
    text_embed = clip.encode_propmt(prompt='a men', do_classifier_free_guidance=True)
    print(text_embed)
    print(text_embed.shape)
