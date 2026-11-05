import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  vus: 10, // 10 concurrent requests
  duration: '30s',
};

const TARGET_URL = __ENV.TARGET_URL || 'http://localhost:5000/classify';

export default function () {
  const payload = JSON.stringify({ text: "Where is my delayed order?" });
  const params = { headers: { 'Content-Type': 'application/json' } };
  const res = http.post(TARGET_URL, payload, params);

  check(res, {
    'status is 200': (r) => r.status === 200,
    'latency is under 5s': (r) => r.timings.duration < 5000,
  });
  sleep(1);
}

