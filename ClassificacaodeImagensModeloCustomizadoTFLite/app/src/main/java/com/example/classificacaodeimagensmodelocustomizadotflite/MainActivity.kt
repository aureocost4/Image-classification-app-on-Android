package com.example.classificacaodeimagensmodelocustomizadotflite

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.ThumbnailUtils
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.Nullable
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.isVisible
import com.example.classificacaodeimagensmodelocustomizadotflite.databinding.ActivityMainBinding
import com.example.classificacaodeimagensmodelocustomizadotflite.ml.Model

import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {

    private var tamanhoImagem: Int = 32

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val botaoCamera = binding.btCamera
        val botaoGaleria = binding.btGaleria

        var solicitaPermissaoCamera = registerForActivityResult(
            ActivityResultContracts.RequestPermission()) { sucesso->
            if (sucesso) abrirCamera()
        }

        botaoCamera.setOnClickListener {
            if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                abrirCamera()
            } else {
                solicitaPermissaoCamera.launch(Manifest.permission.CAMERA)
            }
        }

        botaoGaleria.setOnClickListener {
            val galeriaIntent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(galeriaIntent, 1)
        }
    }

    private fun abrirCamera() {
        val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        startActivityForResult(cameraIntent, 3)
    }

    private fun classificarImagem(imagem: Bitmap?) {
        try {
            val model = Model.newInstance(applicationContext)

            // Cria entradas para referência.
            var inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 32, 32, 3), DataType.FLOAT32)
            var byteBuffer = ByteBuffer.allocateDirect(4 * tamanhoImagem * tamanhoImagem * 3)
            byteBuffer.order(ByteOrder.nativeOrder())

            var arrayValores = IntArray(tamanhoImagem * tamanhoImagem)
            imagem!!.getPixels(arrayValores, 0, imagem.width, 0, 0, imagem.width, imagem.height)
            var pixel = 0
            // Itera sobre cada pixel e extrai os valores R, G e B. Por ultimo, adiciona esses valores ao buffer do tipo byte
            for (i in 0 until (tamanhoImagem)) {
                for (j in 0 until (tamanhoImagem)) {
                    var valor = arrayValores[pixel++] // RGB
                    byteBuffer.putFloat(((`valor` shr 16) and 0xFF) * (1f / 1))
                    byteBuffer.putFloat(((`valor` shr 8) and 0xFF) * (1f / 1))
                    byteBuffer.putFloat((`valor` and 0xFF) * (1f / 1))
                }
            }
            inputFeature0.loadBuffer(byteBuffer)

            // Executa a inferência do modelo e obtém o resultado.
            var outputs = model.process(inputFeature0)
            var outputFeature0 = outputs.outputFeature0AsTensorBuffer

            var inferencias = outputFeature0.floatArray

            // Encontra o index da classe com a maior inferencia
            var posicao = 0
            var maiorInferencia = 0F
            val classes = arrayOf("Apple", "Banana", "Orange")

            for (i in 0 until (inferencias.size)) {
                if (inferencias[i] > maiorInferencia) {
                    maiorInferencia = inferencias[i]
                    posicao = i
                }
            }
            binding.textDescricao.isVisible = true
            binding.textClassificacao.text = classes[posicao]
            
            // Libera recursos do modelo se não for mais usado.
            model.close()
        } catch (e: IOException) {
            Log.d("Erro\n", e.cause.toString())
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, @Nullable data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (resultCode == RESULT_OK) {
            var imagemBitmap: Bitmap? = null

            if (requestCode == 3) {
                imagemBitmap = data?.extras?.get("data") as Bitmap
                val dimensao = imagemBitmap.width.coerceAtMost(imagemBitmap.height)
                imagemBitmap = ThumbnailUtils.extractThumbnail(imagemBitmap, dimensao, dimensao)
                binding.imgClassificacao.setImageBitmap(imagemBitmap)

                imagemBitmap = Bitmap.createScaledBitmap(imagemBitmap, tamanhoImagem, tamanhoImagem, false)
                classificarImagem(imagemBitmap)

            } else {
                val imagemUri = data?.data

                try {
                    imagemBitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, imagemUri)
                } catch (e: IOException) {
                    e.printStackTrace()
                }
                binding.imgClassificacao.setImageBitmap(imagemBitmap)
                imagemBitmap = imagemBitmap?.let { Bitmap.createScaledBitmap(it, tamanhoImagem, tamanhoImagem, false) }
                classificarImagem(imagemBitmap)
            }
        }

    }

}