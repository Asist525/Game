# agent_submission.py

import math
import torch
import kymnasium as kym
from agent import ActorCritic, encode_state_fe_alkkagi, BOARD_W, BOARD_H

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# 공통 베이스: PPO ActorCritic 기반
# ---------------------------------------------------------
class BaseKkaGiAgent(kym.Agent):
    def __init__(self, my_color: int):
        super().__init__()
        self.my_color = my_color   # 0=흑, 1=백
        self.model = ActorCritic().to(DEVICE)

    def act(self, observation, info):
        # turn mismatch 방어 (평가 서버 입장에선 거의 안 쓰일 것)
        if observation["turn"] != self.my_color:
            return {
                "turn": observation["turn"],
                "index": 0,
                "power": 1000.0,
                "angle": 0.0,
            }

        # feature 추출 (여기서 이미 플레이어 시점 y플립 포함)
        state = encode_state_fe_alkkagi(
            observation,
            my_color=self.my_color,
            board_w=BOARD_W,
            board_h=BOARD_H,
        )

        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            stone_logits, power_mean, angle_mean, _ = self.model(state_t)

            stone_idx = stone_logits.argmax(dim=-1).item()
            power_raw = power_mean.item()
            angle_raw = angle_mean.item()

        # --------- 학습 코드와 동일한 스케일링 / 변환 ---------
        # power: [-1,1] -> [500,2000]
        power = 500.0 + (math.tanh(power_raw) + 1.0) * 0.5 * (2000.0 - 500.0)

        # angle_local: [-1,1] -> [-180,180] (플레이어 기준 각도)
        angle_local = 180.0 * math.tanh(angle_raw)

        # 백이면 y축 대칭: angle_global = -angle_local
        if self.my_color == 1:
            angle = -angle_local
        else:
            angle = angle_local
        # -------------------------------------------------------

        return {
            "turn": self.my_color,
            "index": int(stone_idx),
            "power": float(power),
            "angle": float(angle),
        }

    # 평가 코드에서 안 쓸 가능성이 높지만, 제대로 구현해두는 게 좋음
    def save(self, path: str):
        torch.save(self.model.state_dict(), path)


# ---------------------------------------------------------
# 제출용 흑/백 클래스
# ---------------------------------------------------------
class YourBlackAgent(BaseKkaGiAgent):
    def __init__(self):
        super().__init__(my_color=0)

    @classmethod
    def load(cls, path: str) -> 'kym.Agent':
        agent = cls()
        sd = torch.load(path, map_location=DEVICE)
        agent.model.load_state_dict(sd)
        agent.model.eval()
        return agent


class YourWhiteAgent(BaseKkaGiAgent):
    def __init__(self):
        super().__init__(my_color=1)

    @classmethod
    def load(cls, path: str) -> 'kym.Agent':
        agent = cls()
        sd = torch.load(path, map_location=DEVICE)
        agent.model.load_state_dict(sd)
        agent.model.eval()
        return agent
