import time

import gradio as gr
import os
from datetime import datetime

from worldmanager import Manager
from worldsandbox import Sandbox
from command import debug_msg_pool, command_config, command_start
from config import BaseConfig, DevConfig

os.environ['GRADIO_TELEMETRY_ENABLED'] = 'False'

css = """
#msg_textbox {
    color: #FFFFF0
    } 
.container { 
    font-size: 60px; 
    color: #FFFFF0;
}
"""


def remove_before_newline(s):
    # 按照 '\n' 分割字符串
    parts = s.split('\n', 1)
    # 取第二个及之后的所有子字符串，并将它们连接起来
    if len(parts) > 1:
        return parts[1]
    else:
        return ''


class ui_surface:

    def __init__(self):
        # 参数读取
        self.base_config = BaseConfig()
        self.dev_config = DevConfig()
        self.manager = Manager(self.base_config.world_name)
        self.valid_config = self.manager.check(self.base_config.world_name, self.base_config.ai_name)
        self.debug_msg_max_col = 20
        self.debug_msg_len = 0
        self.sandbox = None
        self.query = ''
        self.init_sandbox()

    def init_sandbox(self):
        if self.valid_config:
            self.sandbox = Sandbox(self.base_config.world_name)
            self.sandbox.init_global_agent(self.base_config)

    def user_msg_process(self, query, chat_history):
        self.query = query
        return gr.update(value="", interactive=False), \
            chat_history + [[self.base_config.user_name + '说: ' + query, None]]

    def get_response(self, chat_history):
        if self.query == '':
            yield chat_history
        chat_history[-1][1] = ''
        for chunk_ans in self.sandbox.chat(self.query):
            if chunk_ans is not None:
                chat_history[-1][1] += chunk_ans
                time.sleep(0.05)
                yield chat_history

    def get_response_retry(self, chat_history):
        if self.query == '':
            yield chat_history
        try:
            chat_history[-1][1] = ''
        except IndexError:
            raise IndexError("没有历史提问。")
        for chunk_ans in self.sandbox.chat(command_start + command_config['LIST']['retry']):
            if chunk_ans is not None:
                chat_history[-1][1] += chunk_ans
                time.sleep(0.05)
                yield chat_history

    @staticmethod
    def print_debug_msg(debug_msg):
        debug_res = debug_msg_pool.get_msg()
        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        debug_msg += ('---' + cur_time + '---' + '\n' + debug_res + '\n')
        debug_msg_pool.clear()
        return debug_msg

    def retry_msg_process(self, user_msg, chat_history):
        if self.query == '':
            debug_msg_pool.append_msg("当前没有提问，无法重试。")
            return user_msg, chat_history
        try:
            chat_history[-1] = [self.base_config.user_name + '说: ' + self.query, None]
        except IndexError:
            chat_history = [self.base_config.user_name + '说: ' + self.query, None]
        return gr.update(value="", interactive=False), chat_history

    def start(self):
        with gr.Blocks(css=css) as demo:
            with gr.Tab("chat"):
                chatbot = gr.Chatbot(label='聊天', show_label=True, height=500)

                with gr.Column():
                    with gr.Row():
                        user_msg = gr.Textbox(label='Send a message',
                                              placeholder="按回车提交")

                    with gr.Accordion(label='指令', open=False):
                        with gr.Row():
                            retry_btn = gr.Button("重试")
                            clear_btn = gr.Button("清空")
                            dialog_history_btn = gr.Button("历史对话")
                            context_btn = gr.Button("当前记忆检索内容")
                            prompt_btn = gr.Button("当前提示词内容")
                        with gr.Row():
                            folder_btn = gr.Button("角色文件夹")
                            prompt_file_btn = gr.Button("提示词文件")
                            entity_file_btn = gr.Button("实体文件")
                            history_file_btn = gr.Button("对话历史文件")
                            event_file_btn = gr.Button("事件文件")

                if not self.valid_config:
                    msg_init_value = '身份信息错误，请检查config并重启程序'
                else:
                    msg_init_value = ''

                with gr.Accordion(label='调试信息', open=True):
                    debug_msg_box = gr.Textbox(value=msg_init_value,
                                               show_label=False,
                                               lines=2,
                                               max_lines=10,
                                               interactive=False,
                                               elem_id='msg_textbox',
                                               show_copy_button=True)

                # 用户信息提交
                user_msg.submit(fn=self.user_msg_process, inputs=[user_msg, chatbot],
                                outputs=[user_msg, chatbot],
                                queue=False,
                                show_progress=True).then(
                    fn=self.get_response,
                    inputs=chatbot,
                    outputs=chatbot
                ).then(
                    fn=self.print_debug_msg,
                    inputs=debug_msg_box,
                    outputs=debug_msg_box
                ).then(lambda: gr.update(interactive=True), None, [user_msg], queue=False)

                # ---按钮事件
                # --功能性
                # 重试
                retry_btn.click(fn=self.retry_msg_process, inputs=[user_msg, chatbot],
                                outputs=[user_msg, chatbot],
                                queue=False,
                                show_progress=True).then(
                    fn=self.get_response_retry,
                    inputs=chatbot,
                    outputs=chatbot
                ).then(
                    fn=self.print_debug_msg,
                    inputs=debug_msg_box,
                    outputs=debug_msg_box
                ).then(lambda: gr.update(interactive=True), None, [user_msg], queue=False)
                #
                dialog_history_btn.click(fn=self.show_history, inputs=debug_msg_box, outputs=debug_msg_box)
                prompt_btn.click(fn=self.show_prompt, inputs=debug_msg_box, outputs=debug_msg_box)
                context_btn.click(fn=self.show_context, inputs=debug_msg_box, outputs=debug_msg_box)
                clear_btn.click(lambda: None, None, chatbot, queue=False)

                # --打开文件或文件夹
                folder_btn.click(fn=self.open_folder, inputs=debug_msg_box, outputs=debug_msg_box)
                prompt_file_btn.click(fn=self.open_prompt_file, inputs=debug_msg_box, outputs=debug_msg_box)
                entity_file_btn.click(fn=self.open_entity_file, inputs=debug_msg_box, outputs=debug_msg_box)
                history_file_btn.click(fn=self.open_history_file, inputs=debug_msg_box, outputs=debug_msg_box)
                event_file_btn.click(fn=self.open_event_file, inputs=debug_msg_box, outputs=debug_msg_box)
                # 无效身份信息
                if not self.valid_config:
                    user_msg.interactive = False
                    retry_btn.interactive = False
                    dialog_history_btn.interactive = False
                    prompt_btn.interactive = False
                    context_btn.interactive = False
                    clear_btn.interactive = False
                    folder_btn.interactive = False
                    prompt_file_btn.interactive = False
                    entity_file_btn.interactive = False
                    history_file_btn.interactive = False
                    event_file_btn.interactive = False

            with gr.Tab("config"):
                with gr.Accordion(label='基本设置', open=True):
                    with gr.Accordion(label='单AI对话', open=False):
                        with gr.Row():
                            world_name_text = gr.Textbox(label="世界名字", value=self.base_config.world_name,
                                                         interactive=True)
                            ai_name_text = gr.Textbox(label="AI名字", value=self.base_config.ai_name, interactive=True)
                            user_name_text = gr.Textbox(label="用户名字", value=self.base_config.user_name,
                                                        interactive=True)
                        if self.valid_config:
                            save_single_config_btn_msg = "保存"
                        else:
                            save_single_config_btn_msg = "身份信息不存在，请重新设置并重启程序"
                        save_single_config_btn = gr.Button(save_single_config_btn_msg)
                        save_single_config_btn.click(self.save_single_config,
                                                     inputs=[ai_name_text, world_name_text, user_name_text],
                                                     outputs=save_single_config_btn)

                    with gr.Accordion(label='记忆', open=False):
                        lock_memory_check_box = gr.Checkbox(label='锁定记忆（勾选后对话不会保存到历史文件中）',
                                                            value=self.base_config.lock_memory)
                        history_window_slider = gr.Slider(minimum=0, maximum=20, step=1,
                                                          value=self.base_config.history_window,
                                                          label="初次加载时的上文历史窗口大小", info="范围[0, 20]",
                                                          interactive=True)
                        window_max_token_slider = gr.Slider(minimum=100, maximum=16000, step=10,
                                                            value=self.base_config.window_max_token,
                                                            label="对话窗口最大token值（包括提示词）",
                                                            info="范围[100, 16000]",
                                                            interactive=True)

                        dialog_max_token_slider = gr.Slider(minimum=100, maximum=16000, step=10,
                                                            value=self.base_config.dialog_max_token,
                                                            label="单次对话最大token值", info="范围[100, 16000]",
                                                            interactive=True)

                        token_decrease_size_slider = gr.Slider(minimum=100, maximum=4000,
                                                               step=10,
                                                               value=self.base_config.token_decrease,
                                                               label="超过最大token上限时，历史窗口减少的token大小",
                                                               info="范围[100, 4000]",
                                                               interactive=True)
                        with gr.Row():
                            entity_top_k_slider = gr.Slider(minimum=1, maximum=20, step=1,
                                                            value=self.base_config.entity_top_k,
                                                            label="实体记忆匹配数", info="范围[1, 20]",
                                                            interactive=True)
                            history_top_k_slider = gr.Slider(minimum=1, maximum=20, step=1,
                                                             value=self.base_config.history_top_k,
                                                             label="对话记忆匹配数", info="范围[1, 20]",
                                                             interactive=True)
                            event_top_k_slider = gr.Slider(minimum=1, maximum=20, step=1,
                                                           value=self.base_config.event_top_k,
                                                           label="事件记忆匹配数", info="范围[1, 20]",
                                                           interactive=True)
                        save_memory_btn = gr.Button("保存")
                        save_memory_btn.click(self.save_memory_config,
                                              inputs=[lock_memory_check_box,
                                                      history_window_slider,
                                                      window_max_token_slider,
                                                      dialog_max_token_slider,
                                                      token_decrease_size_slider,
                                                      entity_top_k_slider,
                                                      history_top_k_slider,
                                                      event_top_k_slider],
                                              outputs=save_memory_btn)

                    with gr.Accordion(label='输出', open=False):
                        streaming_check_box = gr.Checkbox(label='流式输出', value=self.base_config.streaming)
                        save_output_btn = gr.Button("保存")
                        save_output_btn.click(self.save_output_config,
                                              inputs=streaming_check_box,
                                              outputs=save_output_btn)

                    with gr.Accordion(label='模型', open=False):
                        temperature_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01,
                                                       value=self.base_config.temperature,
                                                       label="模型温度（越小输出越稳定，越大输出越多元）",
                                                       info="范围[0.0, 1.0]",
                                                       interactive=True)
                        save_model_config_btn = gr.Button("保存")
                        save_model_config_btn.click(self.save_model_config,
                                                    inputs=temperature_slider,
                                                    outputs=save_model_config_btn)

                        gr.Textbox(value="注：以下设置（如更换大模型）请在config.ini更改，且重启程序生效",
                                   show_label=False, interactive=False)
                        with gr.Row():
                            gr.Textbox(label="大模型名",
                                       value=self.base_config.model_name, interactive=False)
                            gr.Textbox(label="设备", value=self.base_config.model_device, interactive=False)
                        gr.Checkbox(label="使用文本转向量模型",
                                    value=self.base_config.use_embedding_model, interactive=False)
                        with gr.Row():
                            gr.Textbox(label="模型名", value=self.base_config.embedding_model, interactive=False)
                            gr.Textbox(label="设备", value=self.base_config.embedding_model_device, interactive=False)
                with gr.Accordion(label='开发设置', open=False):
                    debug_mode_check_box = gr.Checkbox(label='调试模式（运行中会有检索内容输出）',
                                                       value=self.dev_config.DEBUG_MODE,
                                                       interactive=True)
                    answer_extract_enabled_check_box = gr.Checkbox(label='answer_extract_enabled'
                                                                         '（仅提取ai的回答作为记忆，记忆检索中不包含提问）',
                                                                   value=self.dev_config.answer_extract_enabled,
                                                                   interactive=True)
                    fragment_answer_check_box = gr.Checkbox(
                        label='分割ai回答记忆文本（仅answer_extract_enabled为True时有效）',
                        value=self.dev_config.fragment_answer,
                        interactive=True)

                    word_similarity_threshold_slider = gr.Slider(minimum=0.1, maximum=1.0, step=0.01,
                                                                 value=self.dev_config.word_similarity_threshold,
                                                                 label="字词相似度分数阈值",
                                                                 info="范围[0.1, 1.0]，值越大表示相似度越大，"
                                                                      "相似度大的检索内容的会被去掉",
                                                                 interactive=True)
                    update_history_store_step_slider = gr.Slider(minimum=1, maximum=20, step=1,
                                                                 value=self.dev_config.update_history_store_step,
                                                                 label="不锁定记忆（lock_memory为False）的情况下，"
                                                                       "更新载入内存的记忆的频率",
                                                                 info="数值越小更新频率越大，"
                                                                      "建议设置得与运行中的对话平均窗口大小差不多",
                                                                 interactive=True)
                    similarity_comparison_context_window_slider = gr.Slider(minimum=1, maximum=3, step=1,
                                                                            value=self.dev_config.
                                                                            similarity_comparison_context_window,
                                                                            label="用于相似度比较的对话上文窗口",
                                                                            interactive=True)
                    save_dev_config_btn = gr.Button("保存")
                    save_dev_config_btn.click(fn=self.save_dev_config,
                                              inputs=[debug_mode_check_box,
                                                      answer_extract_enabled_check_box,
                                                      fragment_answer_check_box,
                                                      word_similarity_threshold_slider,
                                                      update_history_store_step_slider,
                                                      similarity_comparison_context_window_slider],
                                              outputs=save_dev_config_btn)

        demo.queue()
        demo.launch()

    def save_dev_config(self, debug_mode,
                        answer_extract_enabled,
                        fragment_answer,
                        word_similarity_threshold,
                        update_history_store_step,
                        similarity_comparison_context_window):
        self.dev_config.set_debug_mode(debug_mode)
        self.dev_config.set_answer_extract_enabled(answer_extract_enabled)
        self.dev_config.set_fragment_answer(fragment_answer)
        self.dev_config.set_word_similarity_threshold(word_similarity_threshold)
        self.dev_config.set_update_history_store_step(update_history_store_step)
        self.dev_config.set_similarity_comparison_context_window(similarity_comparison_context_window)
        self.dev_config.save_to_file()
        self.sandbox.cur_agent_reload_dev_config()
        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return gr.update(value="保存 --- " + cur_time + " 已保存 ---")

    def save_single_config(self, ai_name, world_name, user_name):
        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not self.manager.check(world_name, ai_name):
            return gr.update(value=cur_time + " 保存出错：身份不存在")
        self.base_config.set_ai_name(ai_name)
        self.base_config.set_world_name(world_name)
        self.base_config.set_user_name(user_name)
        self.base_config.save_to_file()
        try:
            self.sandbox.cur_agent_set_identity(world_name, ai_name, user_name)
            self.sandbox.cur_agent_reload_config(self.base_config)
        except AttributeError:
            return gr.update(value="保存 --- " + cur_time + " 已保存，请重启程序 ---")
        return gr.update(value="保存 --- " + cur_time + " 已保存 ---")

    def save_memory_config(self, lock_memory,
                           history_window,
                           window_max_token,
                           dialog_max_token,
                           token_decrease,
                           entity_top_k,
                           history_top_k,
                           event_top_k):
        self.base_config.set_lock_memory(lock_memory)
        self.base_config.set_history_window(history_window)
        self.base_config.set_window_max_token(window_max_token)
        self.base_config.set_dialog_max_token(dialog_max_token)
        self.base_config.set_token_decrease(token_decrease)
        self.base_config.set_entity_top_k(entity_top_k)
        self.base_config.set_history_top_k(history_top_k)
        self.base_config.set_event_top_k(event_top_k)
        self.base_config.save_to_file()
        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            self.sandbox.cur_agent_reload_config(self.base_config)
        except AttributeError:
            return gr.update(value="保存 --- " + cur_time + " 已保存，请重启程序 ---")
        return gr.update(value="保存 --- " + cur_time + " 已保存 ---")

    def save_output_config(self, streaming):
        self.base_config.set_streaming(streaming)
        self.base_config.save_to_file()
        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            self.sandbox.cur_agent_reload_config(self.base_config)
        except AttributeError:
            return gr.update(value="保存 --- " + cur_time + " 已保存，请重启程序 ---")
        return gr.update(value="保存 --- " + cur_time + " 已保存 ---")

    def save_model_config(self, temperature):
        self.base_config.set_temperature(temperature)
        self.base_config.save_to_file()
        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            self.sandbox.cur_agent_reload_config(self.base_config)
        except AttributeError:
            return gr.update(value="保存 --- " + cur_time + " 已保存，请重启程序 ---")
        return gr.update(value="保存 --- " + cur_time + " 已保存 ---")

    def show_history(self, debug_msg):
        other_msg = "对话窗口："
        return self.execute_command(command_start + command_config['LIST']['show_temp_history'], debug_msg, other_msg)

    def show_context(self, debug_msg):
        other_msg = "记忆检索内容："
        return self.execute_command(command_start + command_config['LIST']['show_context'], debug_msg, other_msg)

    def show_prompt(self, debug_msg):
        other_msg = "当前提示词："
        return self.execute_command(command_start + command_config['LIST']['show_prompt'], debug_msg, other_msg)

    def open_folder(self, debug_msg):
        other_msg = "打开了角色文件夹"
        return self.execute_command(command_start + command_config['LIST']['folder'], debug_msg, other_msg)

    def open_prompt_file(self, debug_msg):
        other_msg = "打开了提示词文件"
        return self.execute_command(command_start + command_config['LIST']['prompt'], debug_msg, other_msg)

    def open_entity_file(self, debug_msg):
        other_msg = "打开了实体文件"
        return self.execute_command(command_start + command_config['LIST']['entity'], debug_msg, other_msg)

    def open_history_file(self, debug_msg):
        other_msg = "打开了历史文件"
        return self.execute_command(command_start + command_config['LIST']['history'], debug_msg, other_msg)

    def open_event_file(self, debug_msg):
        other_msg = "打开了事件文件"
        return self.execute_command(command_start + command_config['LIST']['event'], debug_msg, other_msg)

    def execute_command(self, command, debug_msg, other_msg):
        if not self.valid_config:
            # 无效设置
            return debug_msg
        res = self.sandbox.chat(command)
        debug_res = debug_msg_pool.get_msg()
        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        debug_msg += ('---' + cur_time + '---' + '\n' + other_msg + '\n' + debug_res + '\n')
        debug_msg_pool.clear()
        return debug_msg


if __name__ == '__main__':
    ui_surface().start()
