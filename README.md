<img src="img/logo.png" alt="Move-se logo" title="Moove-se" align="right" height="60" />

# Instituto Moove-se de Saúde

Este é um **Front-end da empresa fictícia Moove-se** para o **MVP** do curso de **Pós-graduação em Engenharia de Software** da **PUC-Rio**.

Este projeto fornece uma Interface Gráfica para os sistemas de gerenciamento da Moove-se Ltda.

![mvp-front-end-screen](https://github.com/elantunes/mvp-frontend/assets/47639843/b838acb1-cea3-46f5-84b0-379836cb6314)

- Através desta Interface Gráfica podemos gerenciar dados importantes para atividades do dia a dia, como:
- Aluguel (Listagem/Adição/Edição/Exclusão);
- Cliente (Listagem/Adição/Edição/Exclusão);
- Anúncio para Parceiros (Publicação/Excluir Publicação);

## Conteúdo

 * [Pré-requisitos](#pré-requisitos)
 * [Instalação](#instalação)
 * [Como Usar](#como-usar)

## Pré-requisitos

1) Este projeto é apenas uma tela que lê dados obtidos de outras APIs.

    :warning: Para utilização plena da tela, com todos os dados necessários para se trabalhar é preciso antes instalar algumas APIs da Moove-se.

    Visite os links abaixo e siga as instruções dos seus respectivos arquivos `README.md`:

    | Descrição | Link |
    | --- | --- |
    | API de Anúncios | https://github.com/elantunes/mvp-api-anuncios |
    | API Geral | https://github.com/elantunes/mvp-api |

2) A rede precisa ter acesso à URL https://viacep.com.br/ws/cep/json para obter os dados de endereço de um CEP.

    A **viacep.com.br não possui licença de uso e nem requer um cadastro**.

## Instalação

1) Baixe o Repositório.
2) Navegue até a pasta do projeto como no exemplo abaixo:

    ```powershell
    cd ./mvp-frontend
    ```

3) Certifique-se que o serviço do Docker esteja sendo executado e execute o comando abaixo:

    ```powershell
    docker compose up --build -d
    ```

## Como Usar

Acesse no navegador a URL: http://localhost:5000 e aproveite! :rocket:

