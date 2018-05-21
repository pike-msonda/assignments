
// TABS JQUERY
$("#tabs").tabs().css({
    'min-height': '800px',
    'overflow': 'auto'
 });


//FILE UPLOAD:
$('#upload').submit(function(event){
    event.stopPropagation();
    event.preventDefault();

    var file =  document.getElementById('dataFile').files[0];
    var reader = new FileReader();
    reader.readAsText(file, 'UTF-8');
    reader.onload = upload;

    function upload(event){
        var result =  event.target.result;
        var filename = file.name;
        $.post("http://0.0.0.0:8080/upload", {data: result, name: filename},function(data){
            $("#status").text(data);
            $("#status").css('background-color','yellow');
            setTimeout(function() {$("#status").css('background-color','').text("");}, 3000);
            display();
        });
      
    }
    function display(event){
        $.get("http://0.0.0.0:8080/csvhanlder", function(data){
               $(".displayData").empty();
               $(".displayData").append(data);
            });
    }
})

//Neuron Configuration 
$("#neuron").submit(function (event){
    
})