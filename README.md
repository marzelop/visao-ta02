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

## Cálculo realizado

Os 16 cantos são encontrados com `cv2.findChessboardCorners()`, usando
`cv2.findChessboardCornersSB()` como alternativa quando necessário, e refinados
com `cv2.cornerSubPix()`. Os pontos correspondentes do tabuleiro pertencem a um
plano com coordenada Z igual a zero e espaçamento definido por `SQUARE_SIZE`.

`cv2.calibrateCamera()` estima a matriz intrínseca, que contém as distâncias
focais `fx`, `fy` e o ponto principal `cx`, `cy` em pixels, além dos coeficientes
de distorção `k1, k2, p1, p2, k3` e da pose do tabuleiro em cada fotografia.

Para cada imagem utilizada, `cv2.projectPoints()` reprojeta os 16 pontos do
tabuleiro com os parâmetros estimados. O erro informado é a **média das
distâncias euclidianas** entre cada canto detectado e seu ponto reprojetado:

```text
error_i = mean(sqrt((u_detected - u_projected)^2 + (v_detected - v_projected)^2))
```

Os erros são medidos em pixels. Também são informados a média desses erros por
imagem e o RMS retornado pelo OpenCV. O RMS é a raiz da média das distâncias
euclidianas ao quadrado entre todos os pontos; portanto, difere da média das
distâncias. Nenhuma fotografia é excluída automaticamente por erro elevado.

A reprojeção acima serve para medir o erro do ajuste da calibração. O experimento
de projeção com pontos reservados para validação está descrito ao final deste README.

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

Uma nova execução sobrescreve os arquivos de mesmo nome. Os parâmetros podem
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

O script carrega `camera_matrix`, `dist_coeffs` e `image_size` de
`results/<camera_id>/calibration.npz`, sem recalcular a calibração. Usa
as fotografias originais de `images/<camera_id>/` e, quando presentes,
`project_test/<camera_id>/`, na resolução original. A
detecção e o refinamento dos 16 cantos seguem os mesmos métodos, parâmetros e
ordenação usados em `calibrate.py`.

Os pontos 3D são definidos em centímetros, com quadrados de **3,1 cm** e Z igual
a zero. O índice segue a ordem `index = 4 * row + column`, com X variando primeiro:

```text
 0   1   2   3
 4   5   6   7
 8   9  10  11
12  13  14  15
```

A origem é o canto de índice 0, com X e Y seguindo as colunas e linhas detectadas.
**Essas coordenadas pertencem ao sistema de referência do tabuleiro, não a um
sistema global de coordenadas.** Como o padrão é simétrico, a orientação atribuída
pelo detector pode mudar entre fotografias. A pose é estimada individualmente
para cada fotografia e transforma o referencial do tabuleiro para o da câmera.

Os quatro cantos extremos, índices **0, 3, 12 e 15**, são reservados exclusivamente
para validação. Os outros **12 pontos**, distribuídos pelas quatro linhas e
colunas da grade, fornecem as correspondências 3D e 2D conhecidas para uma única
chamada a `cv2.solvePnP()`, com `SOLVEPNP_ITERATIVE`. A divisão é fixa e determinística.
Os pontos de validação não entram na estimação nem em um refinamento posterior
da pose.

`cv2.projectPoints()` projeta apenas os quatro pontos reservados usando a pose,
a matriz intrínseca e a distorção originais. A comparação é feita com seus cantos
detectados na **mesma fotografia original**, sem usar imagens de `undistort.py`.
A validação verifica correspondências que não participaram da estimação da pose.
No conjunto `images/`, essas fotografias também podem ter sido usadas na
calibração; essa separação, por si só, não constitui uma avaliação em fotografias
independentes. As novas fotografias de `project_test/` permitem repetir o
procedimento sem terem participado da calibração intrínseca.

Para cada ponto reservado, o erro é a distância euclidiana em pixels. As métricas
globais são calculadas diretamente de todos os erros individuais da câmera:

```text
error = sqrt((u_observed - u_projected)^2 + (v_observed - v_projected)^2)
mean_error = mean(errors)
RMSE = sqrt(mean(errors^2))
maximum_error = max(errors)
```

O RMSE global não é calculado pela média dos RMSEs por fotografia. O terminal
mostra as contagens, os conjuntos de índices, as métricas globais e o erro médio
de cada imagem. Fotografias ilegíveis, sem tabuleiro detectável ou sem pose válida
são informadas como descartadas. Erros elevados não causam exclusão automática.

Os resultados são gravados em `results/<camera_id>/projections/`:

- Fotografias com os nomes originais: os 12 cantos de pose aparecem como círculos
  verdes; os quatro cantos observados de validação, como anéis azuis; suas projeções,
  como cruzes vermelhas. Linhas amarelas conectam cada observação à sua projeção.
  Os rótulos mostram o índice e o erro em pixels. Os marcadores são ampliados para
  facilitar a visualização; somente o desenho arredonda coordenadas para pixels.
- `points.csv`: quatro linhas por imagem utilizada, uma por ponto de validação,
  com nome da imagem, índice de 0 a 15, X/Y/Z em centímetros, coordenadas 2D
  observadas e projetadas em pixels, e erro euclidiano. As coordenadas mantêm
  precisão de subpixel.
- `summary.txt`: configuração, contagens, índices, métricas globais, médias por
  imagem, descartes e legenda das cores.

Na execução com as calibrações existentes, todas as fotografias disponíveis
tiveram o tabuleiro detectado e a pose estimada, sem descartes. Os erros abaixo
correspondem somente aos quatro pontos reservados de cada imagem:

| Câmera | Imagens utilizadas | Pontos de validação | Erro médio (px) | RMSE (px) | Erro máximo (px) |
| --- | ---: | ---: | ---: | ---: | ---: |
| cam05 | 20/20 | 80 | 2,111742 | 2,562593 | 6,119383 |
| cam1 | 22/22 | 88 | 1,121420 | 1,575436 | 5,121168 |

Os valores completos e os erros por fotografia estão nos arquivos `summary.txt`
e `points.csv` de cada câmera. Novas imagens ou calibrações podem alterar os
resultados de uma execução futura.

### Projeção nas fotografias independentes

Coloque as novas fotografias do tabuleiro em `project_test/<camera_id>/`,
mantendo-as fora de `images/<camera_id>/`. O mesmo comando `python project.py cam1`
processa os dois conjuntos com a calibração existente. Sem argumento,
`python project.py` faz isso para todas as câmeras, incluindo as fotografias
independentes de cada uma quando disponíveis. A detecção, a seleção de
12 pontos de pose e quatro de validação e as métricas são idênticas nos dois
conjuntos. Os novos arquivos não são fornecidos a `calibrate.py`.

As imagens anotadas, `points.csv` e `summary.txt` do conjunto independente são
gravados em `results/<camera_id>/projections/tests/`. O arquivo
`results/<camera_id>/projections/comparison.txt` compara erro médio, RMSE e erro
máximo entre os conjuntos, incluindo as diferenças absolutas e percentuais.
Os erros dos conjuntos são calculados separadamente, sem misturar as amostras.

Essa comparação avalia a projeção em fotografias não usadas na calibração,
embora a pose de cada nova fotografia continue sendo estimada a partir de seus
12 cantos conhecidos. Diferenças de posição, distância e inclinação do tabuleiro
podem influenciar os resultados. Poucas imagens de teste permitem descrever o
resultado observado, mas não sustentam uma conclusão ampla de generalização.

Na execução com as três novas fotografias de cada câmera, todas foram utilizadas
e produziram 12 pontos de validação por câmera, sem descartes. Os arquivos de
calibração foram preservados. Os resultados obtidos foram:

| Câmera | Conjunto | Imagens | Erro médio (px) | RMSE (px) | Erro máximo (px) |
| --- | --- | ---: | ---: | ---: | ---: |
| cam1 | Fotografias da calibração | 22 | 1,121420 | 1,575436 | 5,121168 |
| cam1 | Novas fotografias | 3 | 2,839328 | 4,060751 | 9,162294 |
| cam05 | Fotografias da calibração | 20 | 2,111742 | 2,562593 | 6,119383 |
| cam05 | Novas fotografias | 3 | 2,417905 | 3,060062 | 6,820997 |

O aumento do erro médio foi de 1,717908 px (+153,19%) na `cam1` e de 0,306163 px
(+14,50%) na `cam05`. O RMSE aumentou 157,75% e 19,41%, respectivamente.
Nesta amostra, a diferença foi maior na câmera principal; isso não demonstra
que ela terá pior desempenho em qualquer conjunto independente. As três imagens
de cada câmera têm condições próprias de captura, e a pose continua sendo
estimada individualmente com os mesmos 12 cantos.

Referências: [calibração e projeção no OpenCV](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
e [estimação de pose com solvePnP](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html).
