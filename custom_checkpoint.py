import torch
from tsne_model import CustomResNet

# 예상치 않은 키를 제거할 체크포인트 파일의 리스트
checkpoint_files = ['resnet18_epoch_0.pth', 'resnet18_best.pth', 'resnet18_last.pth']

# 각 체크포인트 파일에 대해 반복합니다.
for checkpoint_file in checkpoint_files:
    # 모델을 정의합니다.
    model = CustomResNet()

    # 모델의 상태 사전을 로드합니다.
    state_dict = torch.load(checkpoint_file, map_location=torch.device('cpu'))

    # 예상치 않은 키를 확인합니다.
    unexpected_keys = [key for key in state_dict.keys() if 'resnet.fc' in key]
    print("Unexpected keys in {}: {}".format(checkpoint_file, unexpected_keys))

    # 예상치 않은 키를 제거합니다.
    for key in unexpected_keys:
        del state_dict[key]

    # 모델의 상태 사전을 업데이트합니다.
    model.load_state_dict(state_dict)

    # 새로운 체크포인트 파일의 이름을 생성합니다.
    new_checkpoint_file = 'new_' + checkpoint_file

    # 새로운 체크포인트 파일을 저장합니다.
    torch.save(model.state_dict(), new_checkpoint_file)

    print("New checkpoint saved as:", new_checkpoint_file)