# -*- coding: utf-8 -*-
"""
Utility functions for path
"""


CHECKPOINT_FILE_NAME = "checkpoint"


def join_oss_path(*segments):
    return "/".join(
        s[:-1] if s.endswith("/") else s
        for s in segments
    )
