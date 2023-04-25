# SFighterAI

[简体中文](README_CN.md) | [English](README.md) | Español

Este proyecto es un agente de IA entrenado con aprendizaje por refuerzo profundo para vencer al jefe final en el juego "Street Fighter II: Special Champion Edition". El agente de IA toma decisiones basándose únicamente en los valores de los píxeles RGB de la pantalla del juego. En el estado guardado proporcionado, el agente logra una tasa de victorias del 100% en la primera ronda del nivel final (ocurre sobreajuste, consulte la sección [Ejecución de pruebas](#ejecución-de-pruebas) para ver la discusión).

### Estructura de archivos

```bash
├───data
├───main
│   ├───logs
│   ├───trained_models
│   └───scripts
├───utils
│   └───scripts
```

Los archivos de configuración del juego se almacenan en la carpeta `data/`, y el código principal del proyecto se encuentra en la carpeta `main/`. Dentro de `main/`, la carpeta `logs/` contiene la salida de la terminal y las curvas de datos que registran el proceso de entrenamiento (se pueden ver con Tensorboard), mientras que la carpeta `trained_models/` contiene los pesos del modelo de diferentes etapas. Estos pesos se pueden utilizar para ejecutar pruebas en `test.py` y observar el rendimiento de las estrategias aprendidas por el agente de IA en diferentes etapas de entrenamiento.

## Guía de ejecución

Este proyecto se basa en el lenguaje de programación Python y utiliza principalmente bibliotecas estándar como [OpenAI Gym Retro](https://retro.readthedocs.io/en/latest/getting_started.html) y [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/). La versión de Python utilizada es 3.8.10, y se recomienda utilizar [Anaconda](https://www.anaconda.com) para configurar el entorno de Python. El siguiente proceso de configuración ha sido probado en Windows 11. A continuación se presentan comandos de la consola/terminal.

### Configuración del entorno

```bash
# Crear un entorno conda llamado StreetFighterAI con la versión de Python 3.8.10
conda create -n StreetFighterAI python=3.8.10
conda activate StreetFighterAI

# Instalar bibliotecas de Python
cd [directorio_padre_del_proyecto]/street-fighter-ai/main
pip install -r requirements.txt

# Ejecutar script para localizar la carpeta del juego gym-retro
cd ..
python .\utils\print_game_lib_folder.py
```

Después de que la consola muestre la ruta de la carpeta, cópiela en el explorador de archivos y navegue hasta la ruta correspondiente. Esta carpeta contiene los archivos de datos del juego "Street Fighter II: Special Champion Edition" dentro de gym-retro, incluyendo el archivo de ROM del juego y los archivos de configuración de datos. Copie los archivos `Champion.Level12.RyuVsBison.state`, `data.json`, `metadata.json` y `scenario.json` de la carpeta `data/` de este proyecto en la carpeta de datos del juego, reemplazando los archivos originales (pueden requerirse privilegios de administrador). El archivo `.state` es un estado guardado para el nivel de dificultad más alto del juego, mientras que los tres archivos `.json` son archivos de configuración de gym-retro que almacenan las direcciones de memoria de la información del juego (este proyecto solo usa [agent_hp] y [enemy_hp] para leer los valores de salud de los personajes en tiempo real).

Para ejecutar el programa, también necesitará el archivo de ROM del juego "Street Fighter II: Special Champion Edition", que no es proporcionado por gym-retro y debe obtenerse legalmente por otros medios. Puede consultar este [enlace](https://wowroms.com/en/roms/sega-genesis-megadrive/street-fighter-ii-special-champion-edition-europe/26496.html).

Una vez que haya obtenido legalmente el archivo de ROM del juego, cópielo a la carpeta de datos del juego de gym-retro mencionada anteriormente y cámbiele el nombre a `rom.md`. En este punto, la configuración del entorno está completa.

Nota 1: Si desea ver la interfaz de usuario de integración gym-retro en el juego para capturar manualmente el estado guardado y buscar variables de memoria, puede usar la interfaz de integración gym-retro ui, copie `data/Gym Retro Integration.exe` a la carpeta de datos del juego de gym-retro mencionada anteriormente (dos niveles superiores, la carpeta `retro/`).

Nota 2: Si desea grabar videos del juego del agente de IA, deberá instalar [ffmpeg](https://ffmpeg.org/).

```bash
conda install ffmpeg
```

### Ejecución de pruebas

Una vez configurado el entorno, puede ejecutar `test.py` en la carpeta `main/` para probar y experimentar el rendimiento del agente de IA en diferentes etapas de entrenamiento.

```bash
cd [directorio_padre_del_proyecto]/street-fighter-ai/main
python test.py
```

Los archivos de peso del modelo se almacenan en la carpeta `main/trained_models/`. El modelo predeterminado utilizado en `test.py` es `ppo_ryu_2500000_steps_updated.zip`, que tiene una buena generalización y es capaz de vencer el último nivel de Street Fighter II: Special Champion Edition. Si desea ver el rendimiento de otros modelos, puede cambiar la variable `model_path` en `test.py` a la ruta de otro archivo de modelo. El rendimiento observado de los modelos en varias etapas de entrenamiento es el siguiente:

* ppo_ryu_2000000_steps_updated: Comenzando a entrar en el estado de sobreajuste, generalizable pero no del todo capaz.
* ppo_ryu_2500000_steps_updated: Acercándose al estado de sobreajuste final, no puede dominar la primera ronda pero parcialmente generalizable. Alta probabilidad de vencer la etapa final.
* ppo_ryu_3000000_steps_updated: Cerca del estado de sobreajuste final, casi domina la primera ronda pero apenas generalizable.
* ppo_ryu_7000000_steps_updated: Sobreajustado, domina la primera ronda pero no es generalizable.

### Entrenamiento del modelo

Si deseas entrenar tu propio modelo, puedes ejecutar `train.py`en la carpeta `main/`.

```bash
cd [directorio_padre_del_proyecto]/street-fighter-ai/main
python train.py
```

### Visualización de las curvas de entrenamiento

El proyecto incluye gráficos de Tensorboard del proceso de entrenamiento. Puedes usar Tensorboard para ver datos detallados. Se recomienda usar el complemento integrado de Tensorboard en VSCode para ver los datos directamente. El método de visualización tradicional se muestra a continuación:

```bash
cd [directorio_padre_del_proyecto]/street-fighter-ai/main
tensorboard --logdir=logs/
```

Abre la dirección predeterminada del servicio Tensorboard `http://localhost:6006/` en tu navegador para ver gráficos interactivos del proceso de entrenamiento.

## Reconocimientos
Este proyecto utiliza bibliotecas de código abierto como [OpenAI Gym Retro](https://retro.readthedocs.io/en/latest/getting_started.html), [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/). Se agradece la contribución de todos los desarrolladores a la comunidad de código abierto.

Dos artículos que tuvieron un impacto significativo en este proyecto:

[1] [DIAMBRA Arena A New Reinforcement Learning Platform for Research and Experimentation](https://arxiv.org/abs/2210.10595)
El resumen valioso de la experiencia en la configuración de hiperparámetros para modelos de aprendizaje profundo por refuerzo en juegos de lucha en este artículo fue de gran ayuda para el proceso de entrenamiento de este proyecto.

[2] [Mitigating Cowardice for Reinforcement Learning](https://ieee-cog.org/2022/assets/papers/paper_111.pdf)
El mecanismo de "decaimiento de penalización" propuesto en este artículo resolvió eficazmente el problema de "cobardía" (siempre evitando a los oponentes y sin atreverse a intentar movimientos de ataque).