"use strict";

const Engine = require('node-uci').Engine;
const axios = require('axios');
const websocket = require('ws');

module.exports = class LichessClient {
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
    sendMove(ws, engine, moves) {
        console.log('Sending a move. Move stack: ' + moves);
        return engine.chain()
        .position('startpos', moves)
        .go({
            movetime: 30000
        })
        .then((data) => {
            console.log(data.info);
            const moveData = JSON.stringify({
                t: 'move',
                d: {
                    u: data.bestmove,
                    b: 1
                }
            });
            console.log('Sending: ' + moveData);
            ws.send(moveData);
        });
    }
    play(game, engine_path) {
        const clientId = Math.random().toString(36).substring(2);
        var socketVersion = 0;
        const socketUrl = 'wss://socket.lichess.org:9029/' + game + '/socket/v2?sri=' + clientId;
        const engine = new Engine(engine_path);
        var moves = [];
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
                    color = 1;
                } else if (response.data.players.white.userId === this.username){
                    color = 0;
                } else {
                    throw new Error("I'm not part of this game.");
                }
                if (color === 1) {
                    console.log('My color is black');
                } else {
                    console.log('My color is white');
                }

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
                    setTimeout(() => {
                        if (socketVersion === color) {
                            // we haven't received any move (socketVersion is still 0)
                            // and we're white, so let's make the first move
                            this.sendMove(ws, engine, moves);
                        }
                    }, 5000);
                });
                ws.on('message', function(data) {
                    console.log('Received: ' + data);
                    const parsed = JSON.parse(data);
                    switch (parsed.t) {
                        case 'move':
                            socketVersion = parsed.v;
                            moves.push(parsed.d.uci);
                            if (parsed.d.ply % 2 === color) {
                                this.sendMove(ws, engine, moves);
                            }
                            break;
                        case 'b':
                            // catch up
                            moves = parsed.d.map(d => d.d.uci);
                            // get the last frame
                            const last = parsed.d[parsed.d.length - 1];
                            socketVersion = last.v;
                            if (last.d.ply % 2  !== color) {
                                this.sendMove(ws, engine, moves);
                            }
                            break;
                    }
                }.bind(this));
                ws.on('error', error => {
                    console.log(error);
                });
            }.bind(this))
            .catch(console.error);
        }.bind(this))
        .catch(console.error);
    }
};
