"use strict";

const Engine = require('node-uci').Engine;
const axios = require('axios');
const websocket = require('ws');

class LichessClient {
    constructor(username, password) {
        this.username = username || '';
        this.password = password || '';
    }
    login() {
        if (!this.username || !this.password) {
            throw new Error('Cannot login. Username or password not set');
        }
        axios.post('https://lichess.org/login', {
            username: this.username,
            password: this.password,
        }, {
            headers: {
                'Accept': 'application/vnd.lichess.v1+json'
            }
        })
        .then(function(response) {
            this.cookie = response.headers['set-cookie'][0].split(';', 1)[0];
            console.log('Login successful');
            console.log(this.cookie);
        }.bind(this))
        .catch(function (error) {
            console.log(error.response);
        });
    }
    play(game, engine_path) {
        const clientId = Math.random().toString(36).substring(2);
        var socketVersion = 0;
        const socketUrl = 'wss://socket.lichess.org:9029/' + game + '/socket/v2?sri=' + clientId;
        const engine = new Engine(engine_path);
        engine
        .init()
        .then(engine => {
            return engine.isready();
        })
        .then(function (engine) {
            console.log('Engine is ready');
            console.log('Fetching game information...');
            axios.get('https://lichess.org/api/game/' + game)
            .then(function(response) {
                var color;
                if (response.data.players.black.userId === this.username) {
                    color = 'black';
                } else if (response.data.players.white.userId === this.username){
                    color = 'white';
                }
                console.log('My color is ' + color);
                return Promise.resolve(color);
            }.bind(this))
            .then(function(color) {
                console.log('Connecting to the WebSocket...');
                const ws = new websocket(socketUrl, null, {
                    headers: {
                        'Cookie': this.cookie,
                    }
                });
                ws.on('open', () => {
                    setInterval(() => {
                        const data = JSON.stringify({t: 'p', v: socketVersion});
                        console.log('Sending: ' + data);
                        ws.send(data);
                    }, 1000);
                });
                ws.on('message', data => {
                    console.log('Received: ' + data);
                    const parsed = JSON.parse(data);
                    switch (parsed.t) {
                        case 'move':
                            break;
                        case 'b':
                            // catch up
                            // get the last frame
                            const last = parsed.d[parsed.d.length - 1];
                            socketVersion = last.v;
                            break;
                    }
                });
                ws.on('error', error => {
                    console.log(error);
                });
            }.bind(this))
            .catch(console.error);
        }.bind(this))
        .catch(console.error);
        //.position('startpos')
        //.go({movetime: 10000})
        //.then(result => {
            //console.log(result);
        //})
        //.catch(error => {
            //console.log(error);
        //});
    }
}

const c = new LichessClient(process.argv[2], process.argv[3]);
c.login();
c.play(process.argv[4], process.argv[5]);
