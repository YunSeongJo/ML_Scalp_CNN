import os
import json
from PIL import Image

def flip_image(image_path, output_dir, flip_lr=True, flip_ud=True):
    image = Image.open(image_path)
    filename = os.path.splitext(os.path.basename(image_path))[0]

    if flip_lr:
        flipped_lr = image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_lr_filename = f"{filename}_lr.jpg"
        flipped_lr_path = os.path.join(output_dir, flipped_lr_filename)
        flipped_lr.save(flipped_lr_path)
        generate_json(image_path, flipped_lr_path)

    if flip_ud:
        flipped_ud = image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_ud_filename = f"{filename}_ud.jpg"
        flipped_ud_path = os.path.join(output_dir, flipped_ud_filename)
        flipped_ud.save(flipped_ud_path)
        generate_json(image_path, flipped_ud_path)

    # flipped_all = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    # flipped_all_filename = f"{filename}_flip_all.jpg"
    # flipped_all_path = os.path.join(output_dir, flipped_all_filename)
    # flipped_all.save(flipped_all_path)
    # generate_json(image_path, flipped_all_path)

def generate_json(original_image_path, flipped_image_path):
    original_filename = os.path.splitext(os.path.basename(original_image_path))[0]
    flipped_filename = os.path.splitext(os.path.basename(flipped_image_path))[0]

    original_json_filename = f"{original_filename}.json"
    original_json_path = os.path.join(json_dir, original_json_filename)

    flipped_json_filename = f"{flipped_filename}.json"
    flipped_json_path = os.path.join(output_json_dir, flipped_json_filename)

    # 원본 이미지에 대응하는 JSON 데이터 로드
    with open(original_json_path, 'r') as original_json_file:
        original_json_data = json.load(original_json_file)

    # 새로운 이미지에 대응하는 JSON 데이터 생성
    flipped_json_data = original_json_data.copy()

    # 새로운 이미지에 대응하는 JSON 데이터 저장
    with open(flipped_json_path, 'w') as flipped_json_file:
        json.dump(flipped_json_data, flipped_json_file)

    # 새로운 JSON 파일 저장
    flipped_json_output_path = os.path.join(output_json_dir, flipped_json_filename)
    with open(flipped_json_output_path, 'w') as flipped_json_output_file:
        json.dump(flipped_json_data, flipped_json_output_file)


input_dir = "./head_image/Training/탈모/"
output_dir = "./head_image/AddData/탈모/"
output_json_dir = "./head_image/AddData/탈모라벨/"
json_dir = "./head_image/Labeldataexample/[라벨]탈모_3.중증/"

os.makedirs(output_dir, exist_ok=True)

# 디렉토리의 모든 이미지 파일에 대해 반복적으로 좌우반전, 상하반전된 이미지 저장 및 JSON 파일 생성 수행
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        image_path = os.path.join(input_dir, filename)
        flip_image(image_path, output_dir, flip_lr=True, flip_ud=False)
        flip_image(image_path, output_dir, flip_lr=False, flip_ud=True)

print("이미지 변환 및 JSON 파일 생성이 완료되었습니다.")