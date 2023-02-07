
function uploadAndFillResponse(event) {
    var response = uploadFile(event);
    document.getElementById("theCode").innerHTML = JSON.stringify(response);
}

function get_filename_extension(filename) {
    return filename.split('.').pop();
}

async function uploadFile(event) {
    
    const file = document.getElementById("fileupload");
    var extension = get_filename_extension(file.files[0].name);
    if(extension != "csv"){
        alert("El archivo introducido tiene que ser de extensión .csv!!!");
    } else {
        document.getElementById("submit_prediction").style.visibility = "hidden";
        alert("Comentarios enviados");
        document.getElementById('response_div').innerHTML = '<p style="text-align: center;"> <i> Los comentarios están siendo procesados; puede tardar un rato. </i>  </p>';
        event.preventDefault();
        const endpoint = "http://localhost:8000/predict";
        const formData = new FormData();

        formData.append("file", file.files[0]);

        var response = await fetch(endpoint,
        {
            method: "post",
            body: formData
            }).then((response) => {
                console.log(response);
                console.log(response == undefined);
                return response.json();
        }).catch(console.error);
        let stringified_response = await JSON.stringify(response);

        var my_html = '';
        my_html += '<h2> Comentarios </h2>';

        var positives = [];
        var negatives = [];
        var neutrals = [];
        
        my_html += '<div class="table-container">';
        my_html += '<table>';
        my_html += '<tr>';
        my_html += '<th style="font-size:16px; text-align: center;"> <b> Comentario </b> </th>';
        my_html += '<th style="font-size:16px; text-align: center;"> <b> Sentimiento </b> </th>';
        my_html += '</tr>';

        for (var i = 0; i < response.length-1; i++) {
            my_html += '<tr>';
            my_html += '<td  style="text-align: center;">' + response[i].comment + '</td>';
            switch (response[i].sentiment) {
                case 0:
                    negatives.push(response[i].comment); 
                    my_html += '<td style="color:red; text-align: center;"> <b> NEGATIVO </b>  </td>';
                    break;
                case 1:
                     neutrals.push(response[i].comment);
                    my_html += '<td style="color:blue; text-align: center;"> <b> NEUTRO </b> </td>';
                    break;
                default:
                    positives.push(response[i].comment);
                    my_html += '<td style="color:green; text-align: center;"> <b> POSITIVO </b> </td>';
            }
            my_html += '</tr>';
        }
        my_html += '</table>';
        my_html += '</div>';

        my_html += '<div class="table-container">'
        my_html += '<table>';
        my_html += '<th  class="table-header" style="font-size:16px; text-align: center;"> Comentarios POSITIVOS </th>';
        positives.forEach(comment => my_html += '<tr> <td style="color:green; text-align: center;">' + comment + '</td> </tr>');
        my_html += '<th style="font-size:16px; text-align: center; text-align: center;"> Comentarios NEUTROS </th>';
        neutrals.forEach(comment => my_html += '<tr> <td style="color:blue; text-align: center;">' + comment + '</td> </tr>');
        my_html += '<th style="font-size:16px; text-align: center;"> Comentarios NEGATIVOS </th>';style="text-align: center"
        negatives.forEach(comment => my_html += '<tr> <td style="color:red; text-align: center;">' + comment + '</td> </tr>');
        my_html += '</table>';
        my_html += '</div>';
        my_html += '<br>';

        total = response[response.length-1].number_of_negatives+response[response.length-1].number_of_neutrals+response[response.length-1].number_of_positives
        my_html += '<h3 style="color:green"> Número de comentarios positivos: ' + response[response.length-1].number_of_positives + ' (' + (response[response.length-1].number_of_positives*100/total).toFixed(3) + ' %)  </h3>';
        my_html += '<h3 style="color:blue"> Número de comentarios neutros: ' + response[response.length-1].number_of_neutrals + ' (' + (response[response.length-1].number_of_neutrals*100/total).toFixed(3) + ' %)  </h3>';
        my_html += '<h3 style="color:red"> Número de comentarios negativos: ' + response[response.length-1].number_of_negatives + ' (' + (response[response.length-1].number_of_negatives*100/total).toFixed(3) + ' %) </h3>';
        switch (response[response.length-1].most_frequent) {
            case 0:
                my_html += '<h2 style="color:red"> Los comentarios <b> NEGATIVOS </b> son los más frecuentes! </h2>';
                break;
            case 1:
                my_html += '<h2 style="color:blue"> Los comentarios <b> NEUTROS </b> son los más frecuentes! </h2>';
                break;
            default:
                my_html += '<h2 style="color:green"> Los comentarios <b> POSITIVOS </b> son los más frecuentes! </h2>';
        }

        document.getElementById('response_div').innerHTML = my_html;
        document.getElementById("submit_prediction").style.visibility = "visible";
    }
}

function hello() {
    alert("hello");
}