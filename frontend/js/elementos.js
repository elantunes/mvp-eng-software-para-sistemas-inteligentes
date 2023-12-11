function ativar_popup_operacao_com_sucesso() {
    const div = $('.popup-operacao-sucesso');
    div.animate({top: '10'}, 500, function() { setTimeout(function() { div.animate({top: '-69'}, 'fast') }, 3000) });
}