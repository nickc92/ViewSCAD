#! /usr/bin/env bash



function testExists {
    EXE=$1
    if ! [ -x "$(command -v $EXE)" ]; then
        echo "No $EXE executable found; please install before installng Jupyter Renderer."
        exit 1;
    fi
}

function install_extensions {
    # Confirm that python, node, & npm are present, or fail with an error.
    testExists python
    testExists node
    testExists npm

    # Register/install Jupyter extensions
    jupyter nbextension enable --py widgetsnbextension
    jupyter labextension install @jupyter-widgets/jupyterlab-manager
    jupyter nbextension install --py --symlink --sys-prefix pythreejs
    jupyter nbextension enable --py --sys-prefix pythreejs

    jupyter lab build

    echo 'SolidPython Jupyter Renderer installed. Start it using `jupyter lab`'
}

install_extensions
