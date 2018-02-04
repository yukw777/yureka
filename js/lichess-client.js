"use strict";

const Engine = require('node-uci').Engine;
const axios = require('axios');
const websocket = require('ws');

module.exports = class LichessClient {
    constructor(username, password, movetime) {
        this.username = username || '';
        this.password = password || '';
        this.movetime = movetime || 30000;
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
    sendMove(ws, engine, moves, initialClock, moveClock) {
        console.log('Sending a move. Move stack: ' + moves);
        const clock = this.parseClock(initialClock, moveClock);
        console.log('Clock: ' + JSON.stringify(clock));
        return engine.chain()
        .position('startpos', moves)
        .go(clock)
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
            console.log('Save the move to moves...');
            moves.push(data.bestmove);
            console.log('moves: ' + moves);
        });
    }
    parseClock(initialClock, moveClock) {
        if (!initialClock || !moveClock) {
            return {
                movetime: this.movetime
            };
        }
        return {
            wtime: Math.floor(moveClock.white * 1000),
            btime: Math.floor(moveClock.black * 1000),
            winc: Math.floor(initialClock.increment * 1000),
            binc: Math.floor(initialClock.increment * 1000),
        };
    }
    play(game, engine_path) {
        const clientId = Math.random().toString(36).substring(2);
        var socketVersion = 0;
        const socketUrl = 'wss://socket.lichess.org:9029/' + game + '/socket/v2?sri=' + clientId;
        const engine = new Engine(engine_path);
        var initialClock;
        var moves = [];
        var pingCount = 0;
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
                    color = 0;
                } else if (response.data.players.white.userId === this.username){
                    color = 1;
                } else {
                    throw new Error("I'm not part of this game.");
                }
                if (color === 0) {
                    console.log('My color is black');
                } else {
                    console.log('My color is white');
                }
                initialClock = response.data.clock;

                return Promise.resolve(color);
            }.bind(this))
            .then(function(color) {
                console.log('Connecting to the WebSocket...');
                const ws = new websocket(socketUrl, null, {
                    headers: {
                        'Cookie': this.cookie,
                    }
                });
                var initialMoveTimeout;
                ws.on('open', () => {
                    setInterval(() => {
                        const data = {t: 'p', v: socketVersion};
                        // supposed to be average lag, but let's just put some bogus number
                        // https://github.com/ornicar/lila/blob/4a9df899ba37b0651f322e4566a3b14fce1915f5/ui/site/src/socket.js#L141
                        if (pingCount % 8 === 2) {
                            data.l = 10;
                        }
                        const strData = JSON.stringify(data);
                        console.log('Sending: ' + strData);
                        ws.send(strData);
                    }, 1000);
                    initialMoveTimeout = setTimeout(() => {
                        if (color === 1) {
                            // we haven't received any move for five seconds
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
                            if (initialMoveTimeout) {
                                console.log('Move received. Clear initialMoveTimeout');
                                clearTimeout(initialMoveTimeout);
                                initialMoveTimeout = null;
                            }
                            socketVersion = parsed.v;
                            if (parsed.d.ply % 2 !== color) {
                                // opponent's move
                                moves.push(parsed.d.uci);
                                console.log('Moves: ' + moves);
                                this.sendMove(ws, engine, moves, initialClock, parsed.d.clock);
                            }
                            break;
                        case 'b':
                            console.log('Catching up. Clear initialMoveTimeout');
                            clearTimeout(initialMoveTimeout);
                            // catch up
                            moves = parsed.d.map(d => d.d.uci);
                            // get the last frame
                            const last = parsed.d[parsed.d.length - 1];
                            socketVersion = last.v;
                            if (last.d.ply % 2  !== color) {
                                this.sendMove(ws, engine, moves, initialClock, last.d.clock);
                            }
                            break;
                        case 'n':
                            pingCount++;
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
