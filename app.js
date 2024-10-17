function mostrarMensaje() {
    // Obtener el archivo subido
    const input = document.getElementById("imageInput");
    const file = input.files[0];
    
    // Si no se ha subido ningún archivo
    if (!file) {
        document.getElementById("mensaje").textContent = "Por favor, sube una imagen.";
        return;
    }

    // Obtener el nombre del archivo
    const nombreArchivo = file.name;

    // Comparar el nombre del archivo y mostrar mensajes correspondientes
    if (nombreArchivo === "1726787332992-er_diagram.png") {
        document.getElementById("mensaje").textContent = "Este es el mensaje para la imagen X.";
    } else if (nombreArchivo === "imagenY.jpg") {
        document.getElementById("mensaje").textContent = "Este es el mensaje para la imagen Y.";
    } else {
        document.getElementById("mensaje").textContent = "Imagen desconocida.";
    }
}
