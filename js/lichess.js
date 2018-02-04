"use strict";

const LichessClient = require('./lichess-client');

const c = new LichessClient(process.argv[2], process.argv[3]);
c.login();
c.play(process.argv[4], process.argv[5]);
