import torch
from torch.utils.model_zoo import load_url
from scipy.special import expit
from PIL import Image
import streamlit as st
import os
import tempfile

from blazeface import FaceExtractor, BlazeFace, VideoReader
from architectures import fornet, weights
from isplutils import utils

# Configuração do dispositivo
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
face_policy = 'scale'
face_size = 224
frames_per_video = 32

# Inicialize o modelo de detecção facial BlazeFace
facedet = BlazeFace().to(device)
facedet.load_weights("blazeface/blazeface.pth")
facedet.load_anchors("blazeface/anchors.npy")
videoreader = VideoReader(verbose=False)
video_read_fn = lambda x: videoreader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn=video_read_fn, facedet=facedet)

# Função para realizar a detecção de deep fakes com base no modelo selecionado
def detect_deep_fake(uploaded_file, selected_model, selected_dataset):
    # Crie um diretório temporário para salvar o arquivo
    temp_dir = tempfile.mkdtemp()

    # Salve o arquivo temporariamente no diretório temporário
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(uploaded_file.read())

    if uploaded_file.type.startswith('image'):
        im = Image.open(temp_file_path)
        im_faces = face_extractor.process_image(img=im)
        im_face = im_faces['faces'][0] if len(im_faces['faces']) > 0 else None

        model_url = weights.weight_url['{:s}_{:s}'.format(selected_model, selected_dataset)]
        net = getattr(fornet, selected_model)().eval().to(device)
        net.load_state_dict(load_url(model_url, map_location=device, check_hash=True))
        transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)

        if im_face is not None:
            faces_t = torch.stack([transf(image=im_face)['image']])

            with torch.no_grad():
                faces_pred = torch.sigmoid(net(faces_t.to(device))).cpu().numpy().flatten()

            avg_score = expit(faces_pred.mean())
            prediction = 'FAKE' if avg_score >= 0.6 else 'REAL'
        else:
            return 'Não foi possível detectar uma face na imagem.'

    elif uploaded_file.type.startswith('video'):
        model_url = weights.weight_url['{:s}_{:s}'.format(selected_model, selected_dataset)]
        net = getattr(fornet, selected_model)().eval().to(device)
        net.load_state_dict(load_url(model_url, map_location=device, check_hash=True))
        transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)

        vid_faces = face_extractor.process_video(temp_file_path)
        faces_t = torch.stack([transf(image=frame['faces'][0])['image'] for frame in vid_faces if len(frame['faces'])])

        with torch.no_grad():
            faces_pred = net(faces_t.to(device)).cpu().numpy().flatten()

        avg_score = expit(faces_pred.mean())
        prediction = 'FAKE' if avg_score >= 0.6 else 'REAL'

    else:
        return 'Tipo de arquivo não suportado.'

    return prediction, avg_score

# Configuração de estilo do Streamlit
st.set_page_config(
    page_title="Detecção de Deep Fakes",
    page_icon="✅",
    layout="wide"
)

# Cabeçalho do aplicativo Streamlit
st.title('Detecção de Deep Fakes')

# Upload de arquivo de imagem ou vídeo
uploaded_file = st.file_uploader('Envie uma imagem ou vídeo', type=['jpg', 'jpeg', 'png', 'mp4'])

if uploaded_file:
    selected_model = st.selectbox('Selecione o modelo', ['EfficientNetB4', 'EfficientNetB4ST','EfficientNetAutoAttB4','EfficientNetAutoAttB4ST'])  # Substitua pelos modelos disponíveis
    selected_dataset = st.selectbox('Selecione o conjunto de dados', ['DFDC', 'FFPP'])  # Substitua pelos conjuntos de dados disponíveis

    if st.button('Detecção'):
        prediction, avg_score = detect_deep_fake(uploaded_file, selected_model, selected_dataset)

        if uploaded_file.type.startswith('image'):
            st.image(uploaded_file, caption='Imagem enviada', use_column_width=True)
        elif uploaded_file.type.startswith('video'):
            st.video(uploaded_file, format='video/mp4')

        st.write(f'Predição: {prediction}')
        st.write(f'Pontuação média: {avg_score}')


