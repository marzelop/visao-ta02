# Calibração, remoção de distorção e projeção com OpenCV

O projeto estima os parâmetros de uma câmera a partir de fotografias de um
tabuleiro com **4 × 4 cantos internos (5 × 5 quadrados)** e utiliza esses parâmetros
para corrigir a distorção das imagens e validar a projeção de pontos do tabuleiro.
Utiliza apenas Python,
NumPy e OpenCV. O pacote `opencv-python-headless` fornece o OpenCV sem interface
gráfica; os diagnósticos são salvos em arquivos.

## Organização por câmera

Cada câmera possui seu próprio diretório de imagens e resultados. Os dados
existentes foram organizados como `cam1`:

```text
images/
└── cam1/                         # Fotografias do tabuleiro
images_test/
└── cam1/                         # Fotografias de teste
project_test/
└── cam1/                         # Tabuleiro em fotografias inéditas
results/
└── cam1/
    ├── calibration.npz
    ├── log.txt
    ├── corners/
    ├── projections/
    ├── undistorted/
    │   ├── alpha1/
    │   └── alpha0/
    ├── comparisons/
    └── tests/
        ├── undistorted/
        │   ├── alpha1/
        │   └── alpha0/
        └── comparisons/
```

Os scripts `calibrate.py`, `undistort.py` e `project.py` recebem o identificador da câmera como
argumento opcional. Sem
argumento, processam **todas as câmeras**, identificadas pelos subdiretórios
imediatos de `images/`, em ordem de nome:

```bash
python calibrate.py
python undistort.py
python project.py
```

Cada câmera utiliza seus próprios parâmetros e diretório de resultados. Execute
a calibração antes da correção para que todas tenham um `calibration.npz`.
Com um identificador informado, apenas essa câmera é processada.
Use um nome de diretório simples, como `cam2`.
Para adicionar outra câmera, coloque suas imagens em `images/cam2/` e,
opcionalmente, em `images_test/cam2/`, e execute:

```bash
python calibrate.py cam2
python undistort.py cam2
```

Os parâmetros e resultados dessa câmera serão gravados apenas em `results/cam2/`.
A correção carrega exclusivamente a calibração da câmera selecionada, sem
utilizar os parâmetros de outra câmera.

## Instalação e execução

Na pasta do projeto, crie um ambiente virtual e instale as dependências:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python calibrate.py cam1
```

Coloque as fotografias JPG em `images/cam1/` (ou `images/<camera_id>/` para outra
câmera). O script lê os arquivos em ordem de nome,
sem redimensioná-los, e também pode ser executado a partir de outra pasta, pois
os caminhos são relativos ao arquivo `calibrate.py`. As imagens de cada câmera
devem ter a mesma resolução; câmeras diferentes podem ter resoluções diferentes.
Fotografias ilegíveis ou sem tabuleiro detectável são
descartadas com uma mensagem. O script exige pelo menos três vistas detectadas;
use diversas posições e inclinações do tabuleiro para estimar os parâmetros.

No início de `calibrate.py`, a constante `SQUARE_SIZE` define o lado de cada
quadrado. A configuração atual é `SQUARE_SIZE = 3.1`, em centímetros. Configure
esse valor conforme o tabuleiro utilizado, mantendo a mesma unidade em todo o
conjunto. Se a medida física for desconhecida, use `1.0` como unidade arbitrária
e ajuste a indicação de unidade no relatório.

## Resultados

O script cria `results/<camera_id>/` automaticamente e imprime o relatório no
terminal. Para `cam1`, os arquivos são:

- `results/cam1/calibration.npz`: matriz intrínseca (`camera_matrix`), distorção
  (`dist_coeffs`), dimensões originais (`image_size`, na ordem largura, altura),
  configuração do tabuleiro (`square_size`, `pattern_size`), RMS (`rms`), erro
  médio (`mean_reprojection_error`), erros por imagem (`reprojection_errors`)
  e os nomes correspondentes (`image_names`).
- `results/cam1/log.txt`: contagens, parâmetros, erros e imagens descartadas
  em formato legível.
- `results/cam1/corners/`: fotografias utilizadas com os 16 cantos desenhados,
  preservando a resolução original e os nomes dos arquivos.

Uma nova execução sobrescreve os arquivos. Os parâmetros podem
ser lidos posteriormente com `numpy.load("results/cam1/calibration.npz")`.

## Remoção de distorção

Com o ambiente virtual ativado e `results/cam1/calibration.npz` disponível, execute:

```bash
python undistort.py cam1
```

Esta etapa reutiliza a calibração existente; não é necessário executar
`calibrate.py` novamente. O script lê `camera_matrix`, `dist_coeffs` e `image_size`
do arquivo NPZ da câmera selecionada e processa todos os JPG de `images/cam1/`
em ordem de nome. Quando `images_test/cam1/` está presente, também processa todos
os JPG dessa pasta com os
mesmos parâmetros de calibração. Os caminhos são relativos ao próprio script.
Fotografias ilegíveis ou com resolução diferente da calibração interrompem a
execução com uma mensagem de erro.

A correção com `cv2.undistort()` compensa as distorções radial e tangencial da
lente usando os parâmetros estimados. Para cada imagem, são calculadas duas
versões com matrizes intrínsecas obtidas por `cv2.getOptimalNewCameraMatrix()`:

- **`alpha=1`**: preserva o campo de visão, permitindo bordas pretas onde não
  existem pixels correspondentes na captura.
- **`alpha=0`**: prioriza pixels válidos, reduzindo o campo de visão e produzindo
  um enquadramento mais fechado para evitar as bordas pretas.

As duas versões mantêm a resolução original. O enquadramento é definido pela
nova matriz intrínseca; o script não aplica recorte nem redimensionamento e não
usa a região de interesse retornada pelo OpenCV para cortar o resultado.

São gerados:

- `results/cam1/undistorted/alpha1/`: todas as imagens corrigidas com `alpha=1`,
  preservando os nomes originais.
- `results/cam1/undistorted/alpha0/`: todas as imagens corrigidas com `alpha=0`,
  preservando os nomes originais.
- `results/cam1/comparisons/`: uma comparação para **cada imagem**, também com o nome
  original. Da esquerda para a direita, os três painéis exibem `Original`,
  `Undistorted (alpha=1)` e `Undistorted (alpha=0)`, em tamanho integral, com os
  rótulos em uma faixa acima. Para imagens de 2992 × 2992 pixels, cada comparação
  tem 8976 × 3092 pixels, incluindo a faixa de 100 pixels de altura.

Para as fotografias de `images_test/cam1/`, a mesma estrutura é criada dentro de
`results/cam1/tests/`:

- `results/cam1/tests/undistorted/alpha1/`: imagens de teste corrigidas com `alpha=1`.
- `results/cam1/tests/undistorted/alpha0/`: imagens de teste corrigidas com `alpha=0`.
- `results/cam1/tests/comparisons/`: comparações com original, `alpha=1` e `alpha=0`,
  na mesma ordem e com os mesmos rótulos das demais comparações.

Os nomes originais e as dimensões das imagens são preservados nos dois conjuntos.
Para gerar todos esses resultados, basta executar `python undistort.py cam1`.

O terminal informa cada imagem processada e as contagens finais. Uma nova execução
sobrescreve os arquivos de mesmo nome nas pastas de saída.

## Projeção 3D → 2D com validação de quatro cantos

Com as calibrações disponíveis, execute `python project.py` para processar todas
as câmeras encontradas nos subdiretórios imediatos de `images/`, em ordem de nome.
O argumento `camera_id` é opcional; para processar apenas uma câmera, execute:

```bash
python project.py cam05
python project.py cam1
```
