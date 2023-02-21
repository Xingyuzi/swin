import torch
model = torch.load("/home/techart/xyz/swin/swin_master/pretrain/pretrain_backup/swin_tiny_patch4_window7_224_copy.pth")
state_dict = model['model']
# model_dict = model.state_dict()

# del state_dict['layers.2.blocks.2.attn.relative_position_bias_table']
# del state_dict['layers.2.blocks.3.attn.relative_position_bias_table']
del state_dict['layers.2.blocks.4.attn.relative_position_bias_table']
del state_dict['layers.2.blocks.5.attn.relative_position_bias_table']
model['model'] = state_dict
torch.save(model, '/home/techart/xyz/swin/swin_master/pretrain/pretrain_backup/change0123.pth')
# model_dict.update(state_dict)

# for key, value in model['model'].items():
#     # print(key)
#     for k in key:
#         if k.startswith('layers.2.blocks.2.attn.relative_position_bias_table'):
#           del state_dict[k]
#         if k.startswith('layers.2.blocks.3.attn.relative_position_bias_table'):
#           del state_dict[k]
#         if k.startswith('layers.2.blocks.4.attn.relative_position_bias_table'):
#           del state_dict[k]
#         if k.startswith('layers.2.blocks.5.attn.relative_position_bias_table'):
#           del state_dict[k]


for key, value in model['model'].items():
    print(key)
