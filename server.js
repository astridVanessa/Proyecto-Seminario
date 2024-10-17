const express = require('express');
const multer = require('multer');
const { exec } = require('child_process');
const path = require('path');
const cors = require('cors');

// Inicializa la app de express
const app = express();
const port = 4001;

// Habilitar CORS
app.use(cors());

// Configuración de Multer para el almacenamiento de archivos
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/');
    },
    filename: (req, file, cb) => {
        const uniqueName = Date.now() + '-' + file.originalname; // Nombre único para evitar conflictos
        cb(null, uniqueName);
    }
});
const upload = multer({ storage: storage });

// Middleware para servir archivos estáticos
app.use(express.static('public'));

// Ruta para procesar la imagen
app.post('/process-image', upload.single('image'), (req, res) => {
    if (!req.file) {
        console.log('Archivo de imagen no recibido');
        return res.status(400).json({ error: 'Archivo de imagen no recibido' });
    }

    console.log('Archivo recibido:', req.file);

    // Ejecutar el script de Python para procesar la imagen
    exec(`python process_image.py uploads/${req.file.filename}`, (error, stdout, stderr) => {
        if (error) {
            console.error('Error al procesar la imagen:', stderr);
            return res.status(500).json({ error: 'Error al procesar la imagen' });
        }

        // Asumiendo que el resultado del script se guarda en stdout
        const finalResult = stdout.trim(); // Asegúrate de obtener el resultado adecuadamente

        // Enviar el resultado final al cliente
        res.json({
            message: 'Imagen cargada y procesada correctamente',
            filename: req.file.filename,
            result: finalResult
        });
    });
});

// Iniciar el servidor
app.listen(port, () => {
    console.log(`Servidor escuchando en http://localhost:${port}`);
});
