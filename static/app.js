var Interval;
function trigger_button()
{
var button_id=document.getElementById("train_button").href;
console.log(button_id);
//console.log(train_para.toString());
if(button_id=="http://127.0.0.1:5000/train_model_fun")
{
document.getElementById("loader_wrapper1").style.display="flex";
//setInterval(trigger_button, 1000);
}



}


