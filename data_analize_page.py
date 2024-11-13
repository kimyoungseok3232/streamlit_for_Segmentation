import json
import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="데이터 분석 및 개별 이미지 확인용", layout="wide")

colors = {
    "Radius": (255, 99, 71),        # Tomate색
    "finger-1": (135, 206, 235),    # 하늘색
    "finger-2": (124, 252, 0),      # 풀색
    "finger-3": (255, 182, 193),    # 분홍색
    "finger-4": (75, 0, 130),       # 인디고색
    "finger-5": (255, 215, 0),      # 금색
    "finger-6": (255, 140, 0),      # 다크 오렌지
    "finger-7": (173, 255, 47),     # 라임 그린
    "finger-8": (0, 206, 209),      # 다크 터쿼이즈
    "finger-9": (148, 0, 211),      # 다크 바이올렛
    "finger-10": (255, 20, 147),    # 딥 핑크
    "finger-11": (0, 191, 255),     # 딥 스카이 블루
    "finger-12": (50, 205, 50),     # 라임색
    "finger-13": (250, 128, 114),   # 연어색
    "finger-14": (0, 0, 205),       # 미디엄 블루
    "finger-15": (32, 178, 170),    # 라이트 시 그린
    "finger-16": (123, 104, 238),   # 미디엄 슬레이트 블루
    "finger-17": (255, 69, 0),      # 레드 오렌지
    "finger-18": (218, 112, 214),   # 오키드
    "finger-19": (220, 20, 60),     # 크림슨
    "Trapezoid": (0, 255, 127),     # 스프링 그린
    "Scaphoid": (70, 130, 180),     # 스틸 블루
    "Trapezium": (255, 228, 181),   # 모카
    "Lunate": (46, 139, 87),        # 시다 그린
    "Triquetrum": (238, 130, 238),  # 바이올렛
    "Hamate": (64, 224, 208),       # 터쿼이즈
    "Capitate": (255, 160, 122),    # 밝은 연어색
    "Ulna": (176, 196, 222),        # 밝은 강철색
    "Pisiform": (72, 61, 139)       # 다크 슬레이트 블루
}

@st.cache_data
def load_data():
    defalult_path = '../data/'

    train_img = defalult_path + 'train/DCM/'
    train_label = defalult_path + 'train/outputs_json/'
    train_img_dir = os.listdir(train_img)
    train_data = {'image_path':[],'label_path':[]}

    for path in train_img_dir:
        img_list = [file for file in os.listdir(train_img+path) if file.endswith(('.jpg', '.png'))]
        for img in img_list:
            train_data['image_path'].append(train_img+path+'/'+img)
            train_data['label_path'].append(train_label+path+'/'+img[:-4]+'.json')

    train = pd.DataFrame(train_data)

    test_img = defalult_path + 'test/DCM/'
    test_img_dir = os.listdir(test_img)
    test_data = {'image_path':[], 'label_path':[]}

    for path in test_img_dir:
        img_list = [file for file in os.listdir(test_img+path) if file.endswith(('.jpg', '.png'))]
        for img in img_list:
            test_data['image_path'].append(test_img+path+'/'+img)
            test_data['label_path'].append(False)

    test = pd.DataFrame(test_data)
    
    return train, test

# 데이터 페이지 단위로 데이터프레임 스플릿
# 입력 - input_df(이미지 데이터), anno_df(박스 그리기 용), rows(한번에 보여줄 데이터 수)
# 출력 - df(이미지 데이터프레임 리스트), df2(박스 그리기 용 데이터프레임 리스트)
@st.cache_data()
def split_frame(input_df, rows):
    df = [input_df.loc[i : i + rows -1, :] for i in range(0, len(input_df), rows)]
    return df

@st.cache_data()
def read_json(label_path):
    with open(label_path) as f:
        data = json.loads(f.read())
    return data

@st.cache_data()
def read_image(path):
    img = cv2.imread(path)
    return img

@st.cache_data()
def rle_to_mask(rle, shape):
    """ RLE 데이터를 디코딩하여 2D 마스크 배열 생성 """
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    rle_pairs = [int(x) for x in rle.split()]
    for idx in range(0, len(rle_pairs), 2):
        start_pixel = rle_pairs[idx]
        run_length = rle_pairs[idx + 1]
        mask[start_pixel:start_pixel + run_length] = 1
    return mask.reshape(shape)

def labeled_image(img, image_path, csv):
    if not csv.empty:
        csv = csv.fillna('')
        img_name = image_path.split('/')[-1]
        label = csv[csv['image_name']==img_name]
        for _, classes, rle in label.values:
            color = np.array(colors[classes])
            mask = rle_to_mask(rle, img.shape[:2])
            if st.session_state['Choosed_annotation'] and classes not in st.session_state['Choosed_annotation']: continue
            elif classes in st.session_state['Choosed_annotation']: img[mask == 1] = color
            else: img[mask == 1] = (img[mask == 1] * 0.5 + color * 0.5).astype(np.uint8)
    return img

def get_train_image(image_path, label_path):
    img = read_image(image_path)
    if st.session_state['show_label']:
        label = read_json(label_path)
        for anno in label['annotations']:
            if st.session_state['Choosed_annotation'] and anno['label'] not in st.session_state['Choosed_annotation']: continue
            cv2.polylines(img, [np.array(anno['points'], dtype=np.int32)], True, colors[anno['label']], 10)
    return img

def get_test_image(image_path, csv):
    label_img = 0
    img = read_image(image_path)
    if st.session_state['show_label']:
        img = labeled_image(img, image_path, csv)
    return img

def show_images(image_pathes, window, mode, csv=pd.DataFrame()):
    cols = window.columns(2)
    for idx,[path,anno] in enumerate(image_pathes.values):
        if idx%2 == 0:
            cols = window.columns(2)
        if mode == 'train':
            img = get_train_image(path, anno)
        if mode == 'test':
            img = get_test_image(path, csv)
        cols[idx%2].image(img)
        cols[idx%2].write(path)

# 데이터 프레임 페이지 단위로 출력
# 입력
## img = train_data or test_data에서 'images'
## anno = train_data or test_data에서 'annotations'
## window = 데이터 프레임 출력할 위치
## type = 이미지 경로
def show_dataframe(data, window, mode, csv=pd.DataFrame()):
    # 가장 윗부분 데이터 정렬할 지 선택, 정렬 시 무엇으로 정렬할지, 오름차순, 내림차순 선택
    top_menu = window.columns(3)
    with top_menu[0]:
        sort = st.radio("Sort Data", options=["Yes", "No"], horizontal=1, index=1, key=[window,1])
    if sort == "Yes":
        with top_menu[1]:
            sort_field = st.selectbox("Sort By", options=data.columns, key=[window,2])
        with top_menu[2]:
            sort_direction = st.radio(
                "Direction", options=["⬆️", "⬇️"], horizontal=True
            )
        data = data.sort_values(
            by=sort_field, ascending=sort_direction == "⬆️", ignore_index=True
        )
    # 데이터 크기 출력
    total_data = data.shape
    with top_menu[0]:
        st.write("data_shape: "+str(total_data))
    con1,con2 = window.columns((1,3))

    # 아래 부분 페이지당 데이터 수, 페이지 선택
    bottom_menu = window.columns((4, 1, 1))
    with bottom_menu[2]:
        batch_size = st.selectbox("Page Size", options=[10, 20, 30], key=[window,3])
    with bottom_menu[1]:
        total_pages = (
            len(data) // batch_size+1 if len(data) % batch_size > 0 else len(data) // batch_size
        )
        current_page = st.number_input(
            "Page", min_value=1, max_value=total_pages, step=1
        )
    with bottom_menu[0]:
        st.markdown(f"Page **{current_page}** of **{total_pages}** ")
    pages = split_frame(data, batch_size)

    con1.dataframe(data=pages[current_page - 1]['image_path'], use_container_width=True)

    show_images(pages[current_page - 1], con2, mode, csv)

def csv_list(output_dir):
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
    return csv_files

def check_same_csv(name, csv):
    i = 1
    while name in csv:
        if i == 1:
            name = name[:-4]+'_'+str(i)+'.csv'
        else:
            name = name[:-6]+'_'+str(i)+'.csv'
        i += 1
    return name

@st.dialog("csv upload")
def upload_csv(csv):
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    # 파일이 업로드되면 처리
    if uploaded_file is not None:
        # Pandas를 사용해 CSV 파일 읽기
        df = pd.read_csv(uploaded_file)
        df = df[['image_name','class','rle']]

        # DataFrame 내용 출력
        st.write("Data Preview:")
        st.dataframe(df)

        input_name = st.text_input("csv 파일 이름 지정", value=uploaded_file.name.replace('.csv', ''))
        if st.button("upload_csv"):
            name = check_same_csv(input_name+'.csv',csv)
            st.write("saved file name: "+name)
            df.to_csv('./output/'+name,index=False)
        if st.button("close"):
            st.rerun()

def csv_to_backup(csv):
    if not os.path.exists('./backup/'):
        os.makedirs('./backup/')
    os.rename('./output/'+csv,'./backup/'+csv)
    st.rerun()

def main():
    if st.sidebar.button("새로고침"):
        st.rerun()
    # 원본데이터 확인 가능 아웃풋도 확인하도록 할 수 있을 듯?
    option = st.sidebar.selectbox("데이터 선택",("이미지 데이터","라벨링","backup"))
    
    # 데이터 로드
    traind, testd = load_data()

    if option == "이미지 데이터":
        with st.sidebar.expander("특정 뼈만 선택"):
            st.session_state['Choosed_annotation'] = []
            for category in colors:
                if st.checkbox(category):
                    st.session_state['Choosed_annotation'].append(category)
        # 트레인 데이터 출력
        choose_data = st.sidebar.selectbox("트레인/테스트", ("train", "test"))
        st.session_state['show_label'] = st.sidebar.checkbox("라벨 표시", value=True)

        if choose_data == "train":
            st.header("트레인 데이터")
            choose_type = st.sidebar.selectbox("시각화 선택", ("이미지 출력"))
            if choose_type == "이미지 출력":
                show_dataframe(traind, st, 'train')

        elif choose_data == "test":
            st.header("테스트 데이터")
            if not os.path.exists('./output/'):
                os.makedirs('./output/')
            csv = csv_list('./output')
            choose_csv = st.sidebar.selectbox("output.csv적용",("안함",)+tuple(csv))

            if choose_csv != "안함":
                ccsv = pd.read_csv('./output/'+choose_csv)
                if st.sidebar.button("현재 csv 백업 폴더로 이동"):
                    csv_to_backup(choose_csv)
            else:
                ccsv = pd.DataFrame()
            show_dataframe(testd, st, 'test', ccsv)
            if st.sidebar.button("새 csv 파일 업로드"):
                upload_csv(csv)

def login(password, auth):
    if password in auth:
        st.session_state['login'] = True
    else:
        st.write('need password')

if 'login' not in st.session_state or st.session_state['login'] == False:
    auth = set(['T7157','T7122','T7262','T7134','T7104','T7201'])
    password = st.sidebar.text_input('password',type='password')
    button = st.sidebar.button('login',on_click=login(password, auth))

elif st.session_state['login'] == True:
    main()