document.addEventListener('DOMContentLoaded', function () {
    const input = document.getElementById('imageInput');

    document.querySelector('button').addEventListener('click', function() {
        const file = input.files[0];

        if (!file) {
            alert('Por favor, selecciona una imagen primero.');
            return;
        }

        const formData = new FormData();
        formData.append('image', file);

        fetch('http://localhost:4001/process-image', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            console.log('Respuesta del servidor:', data);
            
            // Mostrar el mensaje de éxito y el nombre del archivo cargado
            document.getElementById('result').innerText = `Imagen cargada correctamente. Nombre del archivo: ${data.filename}`;

            // Mostrar el resultado del procesamiento
            document.getElementById('result').innerText += `${data.result}`;
        })
        .catch(error => {
            console.error('Error en la solicitud:', error);
            document.getElementById('result').innerText = 'Error al procesar la imagen';
        });
    });
});
