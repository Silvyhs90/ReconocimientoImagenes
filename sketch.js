//variables
var Camara;
var RelacionCamara;
var CartaMensaje;
var Clasificando = false;
var CargandoNeurona = false;
var knn;
var modelo;

//funcion "inicializar"
function setup() {
    //crea un Canvas para dibujar, obteniendo el id en el index.html
  var ObtenerCanva = document.getElementById('micanva');
  var AnchoCanvas = ObtenerCanva.offsetWidth; // saber cuanto es el ancho de esa área
  CartaMensaje = document.getElementById('CartaMensaje');
  CartaMensaje.innerText = "Cargando APP...";
  Camara = createCapture(VIDEO); // activar la camara
  //Camara.size(1280,720);
  Camara.hide();
    //Ajustar el tamaño del Canvas a la resolucion de la Cámara
  RelacionCamara = Camara.height / Camara.width; 
  var AltoCanvas = AnchoCanvas * RelacionCamara; //alto del canvas
  var sketchCanvas = createCanvas(AnchoCanvas, AltoCanvas);
  sketchCanvas.parent("micanva");

  //Se crea un clasificador con Ml5, se usa MobileNet.
  //El MobilNet es una neurona que ya viene pre entrenada a la que se le va a agregar informacion
  //y se le agrega la funcion en este caso ModeloListo imprime en consola que el modelo esta listo
  modelo = ml5.featureExtractor("MobileNet", ModeloListo);
  knn = ml5.KNNClassifier();
    
    //Botones
  BotonesEntrenar = selectAll(".BotonEntrenar"); //funcion de p5
  for (var B = 0; B < BotonesEntrenar.length; B++) {
    BotonesEntrenar[B].mousePressed(PresionandoBoton);
  }

  var TexBoxBoton = select("#TextBoxBoton");
  TexBoxBoton.mousePressed(EntrenarTexBox);
    
  var LimpiarBoton = select("#LimpiarBoton");
  LimpiarBoton.mousePressed(LimpiarNeurona);

  var SalvarBoton =select("#SalvarBoton");
  SalvarBoton.mousePressed(GuardarNeurona);

  var CargarBoton =select("#CargarBoton");
  CargarBoton.mousePressed(CargarNeurona);
  //CargarNeurona();
}

function draw() {
  background("#b2dfdb");

  image(Camara, 0, 0, width, height);
  //Verificar que la neurona tenga informacion que el numero de neuronas sea mayor a 0, que si ya sabe 1 cosa hay que empezar a clasificar, si no , no lo va a hacer, y clasificando se ejecuta una sola vez.
  if (knn.getNumLabels() > 0 && !Clasificando) {
    console.log("Empezar a clasificar");
    setInterval(clasificar, 500); // 500 milisegundos o lo que uno quiera 1 minuto por ejemplo
    Clasificando = true;
  }

    //si el valor cambia, se vuelve a renderizar el canvas
  var RelacionCamara2 = Camara.height / Camara.width;
  if (RelacionCamara != RelacionCamara2) {
    var Ancho = width;
    var Alto = Ancho * RelacionCamara2;
    RelacionCamara = RelacionCamara2;
    //console.log("Cambiando " + Ancho + " - " + Alto);
    resizeCanvas(Ancho, Alto, true);
  }
}


//Funciones
//para ajustar el tamaño de la ventana/camara
function windowResized() {
  var ObtenerCanva = document.getElementById("micanva");
  var Ancho = ObtenerCanva.offsetWidth;
  var Alto = Ancho * RelacionCamara;
  resizeCanvas(Ancho, Alto);
}


function PresionandoBoton() {
  var NombreBoton = this.elt.innerText; //nombre del boton
  console.log("Entrenando con " + NombreBoton);
  EntrenarKnn(NombreBoton);
}

//entrenar neurona, cada vez que tocamos el boton, la informacion se guarda en la memoria de knn
function EntrenarKnn(ObjetoEntrenar) {
  var Imagen = modelo.infer(Camara); //se toma una foto(que en realidad son numeros, informacion en data) y la guarda en la neurona
  knn.addExample(Imagen, ObjetoEntrenar); // se agrega la imagen.
}//la imagen es un ejemplo de este objeto, mientras mas lo entrene mas eficaz es.

function ModeloListo() {
  console.log("Modelo Listo");
  CartaMensaje.innerText="Modelo Listo";
}

//clasificar neurona, que cree la neurona que es lo que entrenamos
//lo que hace es agarrar una imagen actual(la data) y ver si la imagen coincide con la anterior que se guardo
function clasificar() {
  if (Clasificando) {
    var Imagen = modelo.infer(Camara); //inferimos, sacamos informacion de la camara
    knn.classify(Imagen, function(error, result) { //error y resultado que es la respuesta que queremos obtener 
      if (error) {
        console.log("Error en clasificar");
        console.error();
      } else {
        //console.log(result);
        var Etiqueta; //que crea la neurona
        var Confianza; // cuan seguro esta en el resultado
      if(!CargandoNeurona){ //si la neurona se carga por primera vez
          Etiqueta = result.label; // la "etiqueta" con el nombre del objeto
          Confianza = Math.ceil(result.confidencesByLabel[result.label] * 100); // valor de porcentaje
      } else { // si no se carga por primera vez busca en el objeto el key y el value
        Etiquetas = Object.keys(result.confidencesByLabel);
        Valores = Object.values(result.confidencesByLabel);
        var Indice = 0; 
          for (var i = 0; i < Valores.length; i++) { //valor mas alto porque no es string
            if (Valores[i] > Valores[Indice]) {
            Indice = i; //el valor mas alto
          }
        }
        Etiqueta = Etiquetas[Indice];
        Confianza = Math.ceil(Valores[Indice] * 100);
      }
        CartaMensaje.innerText = Etiqueta + " - " + Confianza + "%";
      }
    });
  }
}

//entrenar neurona pero le doy un texto
function EntrenarTexBox() {
  var Imagen = modelo.infer(Camara);
  var EtiquetaTextBox = select("#TextBox").value(); //sacar el valor o texto que se tipeo
  knn.addExample(Imagen, EtiquetaTextBox);
}

//se guarda el modelo en un json para no tener que entrenar siempre que se abre el navegador
function GuardarNeurona() {
  if (Clasificando) {
    console.log("Guardando la neurona");
    knn.save('Neurona'); // se guarda la neurona
  };
}

//se carga la neurona que fue guardada, en formato json tambien
function CargarNeurona() {
  console.log("Cargando una Neurona");
    knn.load("./data/Neurona.json", function() {
   //console.log("Neurona Cargada knn");
   CartaMensaje.innerText = "Neurona cargada de archivo"; // se carga la neurona en el navegador
   CargandoNeurona = true;
  });
}


function LimpiarNeurona() { 
  //console.log("Borrando Neuroona");
  if (Clasificando) {// si esta entrenando
    Clasificando = false;
    clearInterval(clasificar);
    knn.clearAllLabels(); // borrar todo lo que tenga guardado previamente
    CartaMensaje.innerText = "Neurona Borrada";
  }
}
