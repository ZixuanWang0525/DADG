import numpy as np


def load_guidance_schedule(guidance_schedule_path):
    guidance_schedule = np.load(guidance_schedule_path)
    return guidance_schedule


def make_guidance_schedule(warm_up_step, ddim_steps, guidance_duration_steps, guidance_schedule_path):
    guidance_schedule = np.zeros(ddim_steps)
    guidance_schedule[warm_up_step: warm_up_step + guidance_duration_steps] = 1
    np.save(guidance_schedule_path, guidance_schedule)


if __name__ == "__main__":
    guidance_schedule_path = "../baseline/guidance_schedule.npy"
    make_guidance_schedule(warm_up_step=5,
                           ddim_steps=50,
                           guidance_duration_steps=35,
                           guidance_schedule_path=guidance_schedule_path)
    print(load_guidance_schedule(guidance_schedule_path).sum())
